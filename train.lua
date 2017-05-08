--[[
    Train script. Uses torchnet as the framework.

    In order to facilitate the usage of this function, one needs to input a table with the necessary functions and variables to load the training (an the testing [optional]) data. This is achieved by the following input variable 'dataLoadTable', which has to be in the following format:
    {
        (contains the necessary functions and variables to load the necessary data)
        train = {
            getFilename = function(idx) return filename[idx] end,
            getGTBoxes = function(idx) return getGTBoxes[idx] end,
            nfiles = N,
            classLabel = {'car', 'door', ...}
        }
        (* OPTIONAL *)
        test = {
            getFilename = function(idx) return filename[idx] end,
            getGTBoxes = function(idx) return getGTBoxes[idx] end,
            nfiles = N,
            classLabel = {'car', 'door', ...}
        }
    }

    where:
      - getFilename: returns the filename string for the index 'idx'
      - getGTBoxes:  returns a Nx4 FloatTensor with the ground-truth bounding boxes of the objects in the format [x1,y1,x2,y2] for the image with index 'idx'
      - nfiles: total number of files
      - classLabel: list of all class labels/names (it is assumed the labels are sorted)
--]]


--local function train(dataLoadTable, rois, model, modelParameters, opts)
local function train(data_gen, rois, model, modelParameters, opts)

    assert(data_gen, 'Invalid input: dataLoadTable')
    assert(rois, 'Invalid input: rois')
    assert(model, 'Invalid input: model')
    assert(modelParameters, 'Invalid input: modelParameters')

    local tnt = require 'torchnet'
    --local utils = require 'fastrcnn.utils'
    local utils = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/utils/init.lua')
    local modelStorageFn = utils.model.storeModel


    --------------------------------------------------------------------------------
    -- Load configs (data, model, criterion, optimState)
    --------------------------------------------------------------------------------

    local configs = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/configs.lua')
    --local configs = require 'fastrcnn.configs'
    local dataLoadTable = data_gen()
    local opt, modelOut, modelSave, criterion, optimStateFn, nEpochs = configs(model, dataLoadTable, rois, modelParameters, opts or {})
    local lopt = opt

    print('\n==========================')
    print('Optim method: ' .. opt.optMethod)
    print('==========================\n')

    -- set number of iterations
    local nItersTrain = opt.trainIters
    local nItersTest = dataLoadTable.test.nfiles/opt.frcnn_imgs_per_batch

    -- classes
    local classes = utils.table.concatTables({'background'}, dataLoadTable.train.classLabel)

    -- convert modules to a specified tensor type
    local function cast(x) return x:type(opt.dataType) end


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
             --require 'fastrcnn.BatchROISampler'
             paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/init.lua')
             --paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/BatchROISampler.lua')
             --paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/ROIProcessor.lua')
             torch.manualSeed(threadid+opt.manualSeed)
          end,
          closure = function()

              -- data loader/generator
              local data_loader = data_gen()
              local batchprovider = fastrcnn.BatchROISampler(data_loader[mode], rois[mode], modelParameters, opt, mode)

              -- number of iterations per epoch
              local nIters = (mode=='train' and nItersTrain) or nItersTest

              -- setup dataset iterator
              local list_dataset = tnt.ListDataset{
                  list = torch.range(1, nIters):long(),
                  load = function(idx)
                      return batchprovider:getBatch()
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
       train_conf = tnt.ConfusionMeter{k = #classes},
       train_err = tnt.AverageValueMeter(),
       train_cls_err = tnt.AverageValueMeter(),
       train_bbox_err = tnt.AverageValueMeter(),
       train_clerr = tnt.ClassErrorMeter{topk = {1,math.min(#classes, 5)},accuracy=true},

       test_conf = tnt.ConfusionMeter{k = #classes},
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
       test = optim.Logger(paths.concat(opt.savedir,'test.log')),
       train = optim.Logger(paths.concat(opt.savedir,'train.log')),
       full_train = optim.Logger(paths.concat(opt.savedir,'full_train.log')),

       train_conf = optim.Logger(paths.concat(opt.savedir,'train_confusion.log')),
       test_conf = optim.Logger(paths.concat(opt.savedir,'test_confusion.log')),
    }

    loggers.test:setNames{'Test Loss', 'Test acc.', 'Test mAP'}
    loggers.train:setNames{'Train Loss', 'Train acc.'}
    loggers.full_train:setNames{'Train Loss'}
    loggers.train_conf:setNames{'Train confusion matrix'}
    loggers.test_conf:setNames{'Test confusion matrix'}

    loggers.test.showPlot = false
    loggers.train.showPlot = false
    loggers.full_train.showPlot = false
    loggers.train_conf.showPlot = false
    loggers.test_conf.showPlot = false


    -- set up training engine:
    local engine = tnt.OptimEngine()

    engine.hooks.onStart = function(state)
        state.epoch = math.max(opt.epochStart-1, 0)
    end


    engine.hooks.onStartEpoch = function(state)
        state.config = optimStateFn(state.epoch+1)
        print('\n**********************************************')
        print(('*** Starting Train epoch %d/%d, LR=%.0e'):format(state.epoch+1, state.maxepoch, state.config.learningRate))
        print('**********************************************')
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
                xlua.progress(state.t+1, nItersTrain)
            else
                print(('epoch[%d/%d][%d/%d][batch=%d] -  loss: (classification = %2.4f, ' ..
                       ' bbox = %2.4f);   accu: (top-1: %2.2f; top-5: %2.2f);   lr = %.0e')
                       :format( state.epoch+1, state.maxepoch, state.t+1, nItersTrain,
                       state.sample.target[1]:size(1), meters.train_cls_err:value(),
                       meters.train_bbox_err:value(), meters.train_clerr:value{k = 1},
                       meters.train_clerr:value{k = math.min(#classes, 5)},
                       state.config.learningRate))
            end
        else
            xlua.progress(state.t+1, nItersTest)

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
    engine.hooks.onSample = function(state)
        cutorch.synchronize(); collectgarbage();
        utils.table.recursiveCast(samples, state.sample, 'torch.CudaTensor')
        state.sample.input = samples[1]
        state.sample.target = samples[2]
    end


    engine.hooks.onEndEpoch = function(state)
        if state.training then
            local tr = optim.ConfusionMatrix(classes)
            tr.mat = meters.train_conf:value()
            loggers.train_conf:add{tr:__tostring__()} -- output the confusion matrix as a string

            print('\n**Train epoch finished**')
            print(('Train Loss: (total: %0.5f; classification: %0.5f; bbox: %0.5f)  Acc: %0.5f')
                  :format(meters.train_err:value(),
                          meters.train_cls_err:value(),
                          meters.train_bbox_err:value(),
                          meters.train_clerr:value()[1])
            )
            if opt.printConfusion then
                print(tr)
            else
                tr:updateValids();
                print('+ average row correct: ' .. (tr.averageValid*100) .. '%')
                print('+ average rowUcol correct (VOC measure): ' .. (tr.averageUnionValid*100) .. '%')
                print('+ global correct: ' .. (tr.totalValid*100) .. '%')
            end

            -- measure loss and error:
            local tr_loss = meters.train_err:value()
            local tr_accuracy = meters.train_clerr:value()[1]
            loggers.train:add{tr_loss, tr_accuracy}
            meters:reset()

            -- store model
            --modelStorageFn(state.network.modules[1], modelParameters, state.config, state.epoch, state.maxepoch, opt)
            modelStorageFn(modelOut.modules[1], modelSave, modelParameters, state.config, state.epoch, state.maxepoch, opt)
            state.t = 0
        end
    end


    engine.hooks.onEnd = function(state)
        if not state.training then
            local ts = optim.ConfusionMatrix(classes)
            ts.mat = meters.test_conf:value()
            loggers.test_conf:add{ts:__tostring__()} -- output the confusion matrix as a string

            print("Test Loss" , meters.test_err:value())
            print("Accuracy: Top 1%", meters.test_clerr:value()[1])
            print("mean AP:",meters.ap:value():mean())

            if opt.printConfusion then
                print(ts)
            else
                ts:updateValids();
                print('+ average row correct: ' .. (ts.averageValid*100) .. '%')
                print('+ average rowUcol correct (VOC measure): ' .. (ts.averageUnionValid*100) .. '%')
                print('+ global correct: ' .. (ts.totalValid*100) .. '%')
            end

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
        network   = modelOut,
        iterator  = getIterator('train'),
        criterion = criterion,
        optimMethod = optim[opt.optMethod],
        config = optimStateFn(1),
        maxepoch = nEpochs
    }


    --------------------------------------------------------------------------------
    -- Test the model
    --------------------------------------------------------------------------------

    print('\n')
    print('**********************************************')
    print('*** Test the network ')
    print('**********************************************')

    engine:test{
        network   = modelOut,
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

    model = modelOut.modules[1]
end

--------------------------------------------------------------------------------

return train