--[[
    Loads necessary libraries and files for the train script.
]]


local function convert_model_backend(model, opt, is_gpu)
    assert(model)
    assert(opt)
    assert(is_gpu ~= nil)

    if opt.GPU >= 1 and is_gpu then
        print('Running on GPU: num_gpus = [' .. opt.nGPU .. ']')
        require 'cutorch'
        require 'cunn'
        opt.data_type = 'torch.CudaTensor'
        model:cuda()

        -- require cudnn if available
        if pcall(require, 'cudnn') and not opt.disable_cudnn then
            cudnn.convert(model, cudnn):cuda()
            cudnn.benchmark = true
            if opt.cudnn_deterministic then
                model:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
            end
            print('Network has', #model:findModules'cudnn.SpatialConvolution', 'cudnn convolutions')
        end
    else
        print('Running on CPU')
        opt.data_type = 'torch.FloatTensor'

        if pcall(require, 'cudnn') then
            cudnn.convert(model, nn)
        end

        model:float()
    end
    return model
end

------------------------------------------------------------------------------------------------------------

local function LoadConfigs(model, dataLoadTable, rois, modelParameters, opts)

    torch.setdefaulttensortype('torch.FloatTensor')

    -------------------------------------------------------------------------------
    -- Process command line options
    -------------------------------------------------------------------------------

    local opt, optimState, optimStateFn, nEpochs


    local Options = fastrcnn.Options()
    opt = Options:parse(opts or {})

    print('Saving everything to: ' .. opt.savedir)
    os.execute('mkdir -p ' .. opt.savedir)

    if opt.GPU >= 1 then
        require 'cutorch'
        require 'cunn'
        cutorch.setDevice(opt.GPU)
    end

    -- Training hyperparameters
    -- (Some of these aren't relevant for rmsprop which is the optimization we use)
    if not optimState then
        optimState = {
            learningRate = opt.LR,
            learningRateDecay = opt.LRdecay,
            momentum = opt.momentum,
            dampening = 0.0,
            weightDecay = opt.weightDecay
        }
    end

    -- define optim state function ()
    if type(opt.schedule) == 'table' then
        -- setup schedule
        local schedule = {}
        local schedule_id = 0
        for i=1, #opt.schedule do
            table.insert(schedule, {schedule_id+1, schedule_id+opt.schedule[i][1], opt.schedule[i][2], opt.schedule[i][3]})
            schedule_id = schedule_id+opt.schedule[i][1]
        end

        optimStateFn = function(epoch)
            for k, v in pairs(schedule) do
                if v[1] <= epoch and v[2] >= epoch then
                    return {
                        learningRate = v[3],
                        learningRateDecay = opt.LRdecay,
                        momentum = opt.momentum,
                        dampening = 0.0,
                        weightDecay = v[4],
                        end_schedule = (v[2]==epoch and 1) or 0
                    }
                end
            end
            return optimState
        end

        -- determine the maximum number of epochs
        for k, v in pairs(schedule) do
            nEpochs = v[2]
        end

    else
        optimStateFn = function(epoch) return optimState end
    end

    -- Random number seed
    if opt.manualSeed ~= -1 then
        torch.manualSeed(opt.manualSeed)
    else
        torch.seed()
    end


    -------------------------------------------------------------------------------
    -- Setup criterion
    -------------------------------------------------------------------------------

    local criterion = nn.ParallelCriterion()
        :add(nn.CrossEntropyCriterion(), 1)
        :add(nn.BBoxRegressionCriterion(), 1)


    -------------------------------------------------------------------------------
    -- Continue from snapshot
    -------------------------------------------------------------------------------

    opt.curr_save_configs = paths.concat(opt.savedir, 'curr_save_configs.t7')
    if opt.continue then
        if paths.filep(opt.curr_save_configs) then

            -- load snapshot configs
            local confs = torch.load(opt.curr_save_configs)
            opt.bbox_meanstd = confs.bbox_meanstd
            opt.epochStart = confs.epoch + 1

            -- load model from disk
            print('Loading model: ' .. paths.concat(opt.savedir, confs.model_name))
            local modelOut = torch.load(paths.concat(opt.savedir, confs.model_name))[1]

            modelOut = convert_model_backend(modelOut, opt, true)
            criterion:type(opt.data_type)

            return opt, modelOut, criterion, optimStateFn, nEpochs
        end
    end


    -------------------------------------------------------------------------------
    -- Preprocess rois
    -------------------------------------------------------------------------------

    do
        local nSamples = 1000
        print('Compute bbox regression mean/std values over '..nSamples..' train images...')
        local tic = torch.tic()
        local batchprovider = fastrcnn.BatchROISampler(dataLoadTable.train, rois.train, modelParameters, opt, 'train')

        -- compute regression mean/std
        opt.bbox_meanstd = batchprovider:setupData(nSamples)

        print('Done. Elapsed time: ' .. torch.toc(tic))
        print('mean: ', opt.bbox_meanstd.mean)
        print('std: ', opt.bbox_meanstd.std)
    end


    -------------------------------------------------------------------------------
    -- Setup model
    -------------------------------------------------------------------------------

    local modelOut = nn.Sequential()

    -- add mean/std norm
    model:add(nn.ParallelTable()
         :add(nn.Identity())
         :add(nn.BBoxNorm(opt.bbox_meanstd.mean, opt.bbox_meanstd.std)))
    modelOut:add(model)

    -- convert model backend/type
    modelOut = convert_model_backend(modelOut, opt, true)
    criterion:type(opt.data_type)

    if opt.verbose then
        print('Network:')
        print(model)
    end

    -- Save options to experiment directory
    torch.save(paths.concat(opt.savedir, 'options.t7'), opt)
    torch.save(paths.concat(opt.savedir, 'model_parameters.t7'), modelParameters)

    collectgarbage()
    collectgarbage()

    return opt, modelOut, criterion, optimStateFn, nEpochs
end

---------------------------------------------------------------------------------------------------------------------

return LoadConfigs