--[[
    Train script. Uses torchnet as the framework.
--]]


local function train(dataset, rois, model, modelParameters)
  
assert(dataset)
assert(rois)
assert(model)
assert(modelParameters)

local utils = paths.dofile('utils/init.lua')
  
--------------------------------------------------------------------------------
-- Load configs (data, model, criterion, optimState)
--------------------------------------------------------------------------------

local opt, rois_processed, model, criterion, optimStateFn, nEpochs, roi_meanstd = paths.dofile('configs.lua')(model, dataset, rois, utils)
opt.model_params = modelParameters
local lopt = opt

-- save model parameters to experiment directory
torch.save(opt.save .. '/model_parameters.t7', modelParameters)

print('\n==========================')
print('Optim method: ' .. opt.optMethod)
print('==========================\n')

-- load torchnet package
local tnt = require 'torchnet'

-- set model storage function
local modelStorageFn = paths.dofile('utils/store.lua')

-- set number of iterations
local nItersTrain = #rois_processed.train.data
local nItersTest = #rois_processed.test.data

-- classes
local classList = utils.tds_to_table(dataset.data.train.classLabel)
table.insert(classList, 'background')

-- convert modules to a specified tensor type
local function cast(x) return x:type(opt.dataType) end


--------------------------------------------------------------------------------
-- Test data generator
--------------------------------------------------------------------------------
--[[
paths.dofile('data.lua')
local loadData = SetupDataFn('train', rois_processed, opt)
for i=1, 5011 do
  xlua.progress(i,5011)
local data = loadData(i)
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
         
          local nIters = math.ceil(#roi_data/opt.frcnn_imgs_per_batch)
         
          -- setup dataset iterator
          local list_dataset = tnt.ListDataset{  -- replace this by your own dataset
              list = torch.range(1, nIters):long(),
              load = function(idx)
                  return GetBatchSample()
              end
          }
          
          return list_dataset
      end,
   }
end


--------------------------------------------------------------------------------
-- Setup torchnet engine/meters/loggers
--------------------------------------------------------------------------------

local meters = {
   train_conf = tnt.ConfusionMeter{k = opt.nClasses+1},
   train_err = tnt.AverageValueMeter(),
   train_cls_err = tnt.AverageValueMeter(),
   train_bbox_err = tnt.AverageValueMeter(),
   train_clerr = tnt.ClassErrorMeter{topk = {1,5},accuracy=true},
   
   test_conf = tnt.ConfusionMeter{k = opt.nClasses+1},
   test_err = tnt.AverageValueMeter(),
   test_clerr = tnt.ClassErrorMeter{topk = {1},accuracy=true},
   ap = tnt.APMeter(),
}

function meters:reset()
   self.train_conf:reset()
   self.train_err:reset()
   self.train_cls_err:reset()
   self.train_bbox_err:reset()
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
      state.config = optimStateFn(state.epoch+1)
      print('\n**********************************************')
      print(('*** Starting Train epoch %d/%d, LR=%.0e'):format(state.epoch+1, state.maxepoch, state.config.learningRate))
      print('**********************************************')
   else
      print('\n**********************************************')
      print('*** Test the network ')
      print('**********************************************')
   end
end


engine.hooks.onForwardCriterion = function(state)
   if state.training then
      meters.train_conf:add(state.network.output[1],state.sample.target[1])
      meters.train_err:add(state.criterion.output)
      meters.train_cls_err:add(state.criterion.criterions[1].output)
      meters.train_bbox_err:add(state.criterion.criterions[2].output)
       
      meters.train_clerr:add(state.network.output[1],state.sample.target[1])
      loggers.full_train:add{state.criterion.output}
      
      if opt.progressbar then
          xlua.progress((state.t+1)* opt.frcnn_imgs_per_batch, nItersTrain)
      else
          if state.epoch+1 == 2 then
            aqui = 1
          end
            --[[
          print(string.format('epoch[%d/%d][%d/%d][batch=%d] -  loss: (classification = %2.4f,  bbox = %2.4f);     accu: (top-1: %2.2f; top-5: %2.2f);   lr = %.0e',   
          state.epoch+1, state.maxepoch, (state.t+1)* opt.frcnn_imgs_per_batch, nItersTrain, state.sample.target[1]:size(1), state.criterion.criterions[1].output, state.criterion.criterions[2].output, meters.train_clerr:value{k = 1}, meters.train_clerr:value{k = 5}, state.config.learningRate))
        --]]
        --
          print(string.format('epoch[%d/%d][%d/%d][batch=%d] -  loss: (classification = %2.4f,  bbox = %2.4f);     accu: (top-1: %2.2f; top-5: %2.2f);   lr = %.0e',   
          state.epoch+1, state.maxepoch, (state.t+1)* opt.frcnn_imgs_per_batch, nItersTrain, state.sample.target[1]:size(1), meters.train_cls_err:value(), meters.train_bbox_err:value(), meters.train_clerr:value{k = 1}, meters.train_clerr:value{k = 5}, state.config.learningRate))
          --
      end
      
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
local samples = {}
--local inputs = {cast(torch.Tensor()), cast(torch.Tensor())}
--local targets = cast(torch.Tensor())
--if opt.has_bbox_regressor then
--    targets = {cast(torch.Tensor()), {cast(torch.Tensor()), cast(torch.Tensor())}}
--end
---- join tables of tensors function
--local JoinTable = nn.JoinTable(1):float()


engine.hooks.onSample = function(state)
  cutorch.synchronize(); collectgarbage();
  utils.recursiveCast(samples, state.sample, 'torch.CudaTensor')
  state.sample.input = samples[1]
  state.sample.target = samples[2]
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
      modelStorageFn(state.network.modules[1], state.config, state.epoch, state.maxepoch, opt)
      state.t = 0
   end
end


engine.hooks.onEnd = function(state)
   if not state.training then
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
   maxepoch = nEpochs
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