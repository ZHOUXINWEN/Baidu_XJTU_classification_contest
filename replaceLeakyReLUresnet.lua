--[[
 adapt and update ResNeXt 101 60x4d for this question
--]]
require 'nn';
require 'cunn';
require 'cudnn';   
require 'cutorch'

pretrainDir = '/data/DeepGrasp/pretrain/'

cutorch.setDevice(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- pretrain Model is downloaded from https://s3.amazonaws.com/resnext/imagenet_models/resnext_101_64x4d.t7
loadedNet = torch.load(pretrainDir..'resnext_101_64x4d.t7')

-- replace 7x7 SpatialAveragePooling with SpatialAdaptiveAveragePooling
-- replace 2048->1000 fully connected layer with 2048->100 fully connected layer
loadedNet:remove(11)
loadedNet:remove(10)
loadedNet:remove(9)

loadedNet:add(nn.SpatialAdaptiveAveragePooling(1,1))
loadedNet:add(nn.View(2048))
loadedNet:add(nn.Linear(2048,100))

-- replace ReLU with LeakyReLU 
loadedNet:replace(function(module)
   if torch.typename(module) == 'cudnn.ReLU' then
      return nn.LeakyReLU(0.1)
   else
      return module
   end
end)

print(loadedNet)
--BNInit('cudnn.SpatialBatchNormalization')
torch.save(pretrainDir..'resnext_101_64x4d_Leaky01.t7', loadedNet)
--model:cuda():forward(input)--]]


