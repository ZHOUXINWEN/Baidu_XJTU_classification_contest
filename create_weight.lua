--[[
this file is used to create weight for cross entropy loss 
the weight file is save as .t7 file and will be loaded in main.lua
--]]

require 'gnuplot'

trainGT = torch.load('/home/svc3/Baidu/cache/AllGT.t7')
trainPath = torch.load('/home/svc3/Baidu/cache/AllPath.t7')

trainsta = torch.Tensor(100):fill(0)
x = torch.range(1,100)

for i = 1,#trainPath do
	trainsta[trainGT[trainPath[i]]] = trainsta[trainGT[trainPath[i]]] +1
end
max, indice = torch.max(trainsta, 1)
maxTensor = torch.Tensor(100):fill(max[1])
weight = torch.cdiv(maxTensor, trainsta)

for i = 1,100 do
	print(i,weight[i],trainsta[i])
end
torch.save('all_weight.t7',weight)
print(torch.min(weight, 1))
--torch.save('trainsta.t7',trainsta)

--[[gnuplot.pngfigure('trainSta.png')
gnuplot.axis{1,100,0,''}
gnuplot.plot(x,trainsta)
gnuplot.close()
--]]
