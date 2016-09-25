--[[
    Train script. Uses torchnet as the framework.
--]]


local function train(dataset, rois, model, modelParameters)
  
assert(dataset)
assert(rois)
assert(model)
assert(modelParameters)

local utils = paths.dofile('util/utils.lua')
  
--------------------------------------------------------------------------------
-- Load configs (data, model, criterion, optimState)
--------------------------------------------------------------------------------

local opt, rois_processed, model, criterion, optimStateFn = paths.dofile('configs.lua')(model, dataset, rois)
opt.model_params = modelParameters
local lopt = opt

-- save model parameters to experiment directory
torch.save(opt.save .. '/model_parameters.t7', modelParameters)

-- load torchnet package
local tnt = require 'torchnet'

-- set model storage function
local modelStorageFn = paths.dofile('util/store.lua')

-- set number of iterations
local nItersTrain = #rois_processed.train.data
local nItersTest = #rois_processed.test.data

-- classes
local classList = utils.tds_to_table(dataset.data.train.classLabel)
table.insert(classList, 'background')

-- convert modules to a specified tensor type
local function cast(x) return x:type(opt.dataType) end

-- set means, stds for the regressor layer normalization
local roi_means = cast(torch.cat(rois_processed.train.means:view(-1,1), torch.ones(4,1), 1))
local roi_stds = cast(torch.cat(rois_processed.train.stds:view(-1,1), torch.ones(4,1), 1))


--------------------------------------------------------------------------------
-- Test data generator
--------------------------------------------------------------------------------
--[[
paths.dofile('data.lua')
local loadData = SetupDataFn('train', rois_processed, opt)
for i=1, 5011 do
  xlua.progress(i,5011)
local img, rois, labels, bbox_targets, loss_weights = loadData(i)
end
print(rois:min())
os.exit()
local conv = convertIdxRoiFn(rois)
print('aqui')
--]]

--------------------------------------------------------------------------------
-- Setup data generator
--------------------------------------------------------------------------------

local function getIterator(mode)
   return tnt.ParallelDatasetIterator{
      nthread = opt.nThreads,
      init    = function(threadid) 
                  require 'torch'
                  require 'torchnet'
                  opt = lopt
                  paths.dofile('data.lua')
                  torch.manualSeed(threadid+opt.manualSeed)
                end,
      closure = function()
         
         local roi_data = rois_processed[mode].data
         local loadData = SetupDataFn(mode, rois_processed, opt)
         
         local nIters = #roi_data
         local batchSize = (mode == 'train' and opt.frcnn_imgs_per_batch) or (mode == 'test' and opt.nGPU)
         
         -- setup dataset iterator
         local list_dataset = tnt.ListDataset{  -- replace this by your own dataset
            list = torch.range(1, nIters):long(),
            load = function(idx)
                local img, rois, labels, bbox_targets, loss_weights = loadData(idx)
                    return {
                        input = {img, rois},
                        target = {labels, {bbox_targets, loss_weights}}
                    }
            end
          }
          
          return list_dataset
            :shuffle()
            :batch(batchSize, 'include-last')
      end,
   }
end


--------------------------------------------------------------------------------
-- Setup torchnet engine/meters/loggers
--------------------------------------------------------------------------------

local meters = {
   train_conf = tnt.ConfusionMeter{k = opt.nClasses+1},
   train_err = tnt.AverageValueMeter(),
   train_clerr = tnt.ClassErrorMeter{topk = {1},accuracy=true},
   
   test_conf = tnt.ConfusionMeter{k = opt.nClasses+1},
   test_err = tnt.AverageValueMeter(),
   test_clerr = tnt.ClassErrorMeter{topk = {1},accuracy=true},
   ap = tnt.APMeter(),
}

function meters:reset()
   self.train_conf:reset()
   self.train_err:reset()
   self.train_clerr:reset()
   self.test_conf:reset()
   self.test_err:reset()
   self.test_clerr:reset()
   self.ap:reset()
end

local loggers = {
   test = optim.Logger(paths.concat(opt.save,'test.log')),
   train = optim.Logger(paths.concat(opt.save,'train.log')),
   full_train = optim.Logger(paths.concat(opt.save,'full_train.log')),
}

loggers.test:setNames{'Test Loss', 'Test acc.', 'Test mAP'}
loggers.train:setNames{'Train Loss', 'Train acc.'}
loggers.full_train:setNames{'Train Loss'}

loggers.test.showPlot = false
loggers.train.showPlot = false
loggers.full_train.showPlot = false


-- set up training engine:
local engine = tnt.OptimEngine()

engine.hooks.onStartEpoch = function(state)
   if state.training then
      print('\n**********************************************')
      print(('*** Starting Train epoch %d/%d'):format(state.epoch+1, state.maxepoch))
      print('**********************************************')
      state.config = optimStateFn(state.epoch+1)
   else
      print('\n**********************************************')
      print('*** Test the network ')
      print('**********************************************')
   end
end


engine.hooks.onForwardCriterion = function(state)
   if state.training then
      xlua.progress((state.t+1)* opt.frcnn_imgs_per_batch, nItersTrain)
      
      meters.train_conf:add(state.network.output[1],state.sample.target[1])
      meters.train_err:add(state.criterion.output)
      meters.train_clerr:add(state.network.output[1],state.sample.target[1])
      loggers.full_train:add{state.criterion.output}
      
   else
      xlua.progress(state.t*opt.nGPU, nItersTest)
      
      meters.test_conf:add(state.network.output[1],state.sample.target[1])
      meters.test_err:add(state.criterion.output)
      meters.test_clerr:add(state.network.output[1],state.sample.target[1])
      
      local tar = torch.ByteTensor(#state.network.output[1]):fill(0)
      for k=1,state.sample.target[1]:size(1) do
         tar[k][state.sample.target[1][k]]=1
      end
      meters.ap:add(state.network.output[1],tar)
      
   end
end

-- copy sample to GPU buffer:
local inputs = {cast(torch.Tensor()), cast(torch.Tensor())}
local targets = cast(torch.Tensor())
if opt.train_bbox_regressor then
    targets = {cast(torch.Tensor()), {cast(torch.Tensor()), cast(torch.Tensor())}}
end

-- join tables of tensors function
local JoinTable = nn.JoinTable(1):float()

-- correct image idx sent to the roi boxes
local function convertIdxRoiFn(boxes)
  local out = boxes[{{},{1}}]:clone()
  local idxID, idxVal = 0,0
  out:apply(function(idx) 
      if idxID~=idx then 
          idxID=idx
          idxVal=idxVal+1
          return idxVal 
      else 
          return idxVal 
      end 
  end)
  boxes[{{},{1}}]:copy(out)
  return boxes
end

engine.hooks.onSample = function(state)

   -- resize input image tensor
   if state.training then
      inputs[1]:resize(opt.frcnn_imgs_per_batch,3,opt.frcnn_max_size, opt.frcnn_max_size)
   else
      inputs[1]:resize(opt.nGPU,3,opt.frcnn_test_max_size, opt.frcnn_test_max_size)
   end
   
   -- merge data from all batches into single tensors
   -- boxes
   local boxes = {}
   for i=1, #state.sample.input do
     if opt.nGPU > 1 then
        table.insert(boxes, torch.cat(torch.zeros(state.sample.input[i][2]:size(1)):fill(1), state.sample.input[i][2], 2))
     else
        table.insert(boxes, torch.cat(torch.zeros(state.sample.input[i][2]:size(1)):fill(i), state.sample.input[i][2], 2))
     end
   end
   -- join table of tensors into a single tensor
   boxes = JoinTable:forward(boxes):clone()
   
   for i=1, #state.sample.input do
      inputs[1][{{i},{},{1,state.sample.input[i][1]:size(2)},{1,state.sample.input[i][1]:size(3)}}]:copy(state.sample.input[i][1])
   end
   inputs[2]:resize(boxes:size()):copy(boxes) -- img idxs need to be corrected
   
   if opt.train_bbox_regressor then
      local labels, bbox_targets, loss_weights = {}, {}, {}
      for i=1, #state.sample.input do
         table.insert(labels, state.sample.target[i][1])
         table.insert(bbox_targets, state.sample.target[i][2][1])
         table.insert(loss_weights, state.sample.target[i][2][2])
      end
      -- join into tensors
      labels = JoinTable:forward(labels):clone()
      bbox_targets = JoinTable:forward(bbox_targets):clone()
      loss_weights = JoinTable:forward(loss_weights)
      
      targets[1]:resize(labels:size()):copy(labels)
      targets[2][1]:resize(bbox_targets:size()):copy(bbox_targets)
      targets[2][2]:resize(loss_weights:size()):copy(loss_weights)
   else
      targets:resize(state.sample.target:size()):copy(state.sample.target)
   end
   
   state.sample.input  = inputs
   state.sample.target = targets
end


engine.hooks.onEndEpoch = function(state)
   if state.training then
      print(('Train Loss: %0.5f; Acc: %0.5f'):format(meters.train_err:value(),  meters.train_clerr:value()[1]))
      --local tr = optim.ConfusionMatrix(opt.nClasses+1)
      local tr = optim.ConfusionMatrix(classList)
      tr.mat = meters.train_conf:value()
      print(tr)
      
      -- measure loss and error:
      local tr_loss = meters.train_err:value()
      local tr_accuracy = meters.train_clerr:value()[1]
      loggers.train:add{tr_loss, tr_accuracy}
      meters:reset()
      
      -- store model
      --torch.save(paths.concat(opt.save, 'model_fin.t7'),state.network)
      modelStorageFn(state.network.modules[1], state.config, state.epoch, roi_means, roi_stds, opt)
      state.t = 0
   end
end


engine.hooks.onEnd = function(state)
   if not state.training then
      --local ts = optim.ConfusionMatrix(opt.nClasses+1)
      local ts = optim.ConfusionMatrix(classList)
      ts.mat = meters.test_conf:value()
      print(ts)
      print("Test Loss" , meters.test_err:value())
      print("Accuracy: Top 1%", meters.test_clerr:value()[1])
      print("mean AP:",meters.ap:value():mean())
      
      -- measure loss and error:
      local ts_loss = meters.test_err:value()
      local ts_accuracy = meters.test_clerr:value()[1]
      loggers.test:add{ts_loss, ts_accuracy, meters.ap:value():mean()}
      meters:reset()
      
      state.t = 0
   end
end


--------------------------------------------------------------------------------
-- Train the model
--------------------------------------------------------------------------------

engine:train{
   network   = model,
   iterator  = getIterator('train'),
   criterion = criterion,
   optimMethod = optim[opt.optMethod],
   config = optimStateFn(1),
   maxepoch = opt.nEpochs
}


--------------------------------------------------------------------------------
-- Test the model
--------------------------------------------------------------------------------

engine:test{
   network   = model,
   iterator  = getIterator('test'),
   criterion = criterion
}


--------------------------------------------------------------------------------
-- Plot loggers into disk
--------------------------------------------------------------------------------

print('==> Plotting final loggers graphs into disk... ')
loggers.test:style{'+-', '+-'}; loggers.test:plot()
loggers.train:style{'+-', '+-'}; loggers.train:plot()
loggers.full_train:style{'-', '-'}; loggers.full_train:plot()
print('Training script complete.')

end

--------------------------------------------------------------------------------

return train