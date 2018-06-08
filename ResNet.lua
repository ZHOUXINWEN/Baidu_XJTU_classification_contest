require 'nn';
require 'cunn';
require 'cudnn';
optnet = require 'optnet'
nninit = require 'nninit'
torch.setdefaulttensortype('torch.CudaTensor')
--[[
classe = 18
con = { 
   bpc = {6},
   imgshape = 336
}--]]


local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel()
   local shortcutType = 'B'
   local iChannels

  -- xavier initilization and bias zero
  local function xconv(ic,oc,kw,kh,sw,sh,pw,ph,type,dw,dh,relu)
    local conv
    use_relu = relu
    if type == 'N' then
      conv = cudnn.SpatialConvolution(ic, oc, kw, kh, sw, sh, pw, ph):init('weight', nninit.xavier, {dist='uniform', gain=1.1})
    elseif type == 'D' then
      local karnel = torch.randn(oc, ic, kw, kh)
      conv = nn.SpatialDilatedConvolution(ic, oc, kw, kh, sw, sh, pw, ph, pw, ph)
      nninit.xavier(nn.SpatialConvolution(ic, oc, kw, kh, sw, sh, pw, ph), karnel, {dist='uniform', gain=1.1})
      conv.weight:copy(karnel)
    end
    if cudnn.version >= 4000 then
      conv.bias = nil
      conv.gradBias = nil
    else
      conv.bias:zero()
    end
    if use_relu then
      return nn.Sequential():add(conv):add(nn.SpatialBatchNormalization(oc)):add(ReLU)
    else
      return conv
    end
  end

   local function shortcut(nInputPlane, nOutputPlane, stride)
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
         -- 1x1 convolution
         return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
            :add(SBatchNorm(nOutputPlane))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(2)
               :add(nn.Identity())
               :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
   end

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1))
      s:add(SBatchNorm(n))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   -- The bottleneck residual layer for 50, 101, and 152 layer networks
   local function bottleneck(n, stride)
      local nInputPlane = iChannels
      iChannels = n * 4

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n*4,1,1,1,1,0,0))
      s:add(SBatchNorm(n * 4))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1))
      end
      return s
   end

  local main =nn.Sequential() 
  local branch = nn.Sequential()
  if pretrain ~= nil then
        main = torch.load(pretrain)
        print('load pretrained net from '..pretrain )
	--print(main)
  else   
    iChannels = 64
    main = nn.Sequential()
    main:add(Convolution(3,64,7,7,2,2,3,3))   -- 224*224 -> 112*112    2
    main:add(SBatchNorm(64))
    main:add(ReLU(true))
    main:add(Max(3,3,2,2,1,1))                -- 112*112 -> 56*56      4 
    main:add(layer(bottleneck, 64, 3))         
    main:add(layer(bottleneck, 128, 4, 2))    -- 56*56 -> 28*28        8
    main:add(layer(bottleneck, 256, 23, 2))    -- 28*28 -> 14*14        16
    main:add(layer(bottleneck, 512, 3, 2))    -- 14*14 -> 7*7
    main:add(Avg(7, 7, 1, 1))                 
    main:add(nn.View(2048):setNumInputDims(3))
    main:add(nn.Linear(2048, 100))
  end
    --main:add(cudnn.SpatialConvolution(2048, 512, 1, 1, 1, 1)) -- for 50 or deeper 
    --main:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

    main = main:cuda()

    --[[local inp = torch.randn(1, 3, 336, 336):cuda()
    local opts = {inplace=true, mode='training'}
    optnet.optimizeMemory(main, inp, opts)--]]
    return main
end

return createModel
