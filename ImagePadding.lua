require 'pl'
require 'image'
require 'optim'
require 'cutorch'
require 'gnuplot'
require 'nn'
require 'cudnn'
--local matio = require 'matio'
--cutorch.setDevice(2)
torch.setdefaulttensortype('torch.FloatTensor')
local per_start_p = {}
for data in io.lines('/home/zxw/Baidu/datasets/test/imagename.txt') do

	local img = image.load('/home/zxw/Baidu/datasets/test/'..data)
        local n = img:size(2)
	local m = img:size(3)

        print(data)
        -- print('m ', m, 'n ', n)
	if m == n then
		paddedImage = img
        elseif m>n then
        	paddedImage = torch.Tensor(3,m,m):fill(0)
                local startp = math.ceil(m/2-n/2)
		local endp =  startp + n - 1 
                -- print((m/2-n/2), (m/2+n/2))
                -- print('startp ', startp, 'endp ', endp)
                paddedImage[{{},{startp, endp },{}}] = img		
	else
        	paddedImage = torch.Tensor(3,n,n):fill(0)
                local startp = math.ceil(n/2-m/2)
		local endp =  startp + m - 1
                -- print((n/2-m/2), (n/2+m/2))
                -- print('startp ', startp, 'endp ', endp)
                paddedImage[{{},{},{startp, endp }}] = img				
	end
	per_start_p[data] = torch.Tensor({m, n, startp, endp}) 
	image.save('/home/zxw/Baidu/datasets/Test_PaddedImage/'..data,paddedImage)
end
--torch.save('per_start_p.t7', per_start_p)


