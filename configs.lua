--[[
    Loads necessary libraries and files for the train script.
]]


local function LoadConfigs(model, dataset, rois, utils)

  -------------------------------------------------------------------------------
  -- Load necessary libraries and files
  -------------------------------------------------------------------------------

  local roisFn = paths.dofile('rois.lua')
  paths.dofile('WeightedSmoothL1Criterion.lua')
  torch.setdefaulttensortype('torch.FloatTensor')


  -------------------------------------------------------------------------------
  -- Process command line options
  -------------------------------------------------------------------------------

  local opt, optimState, optimStateFn, nEpochs
  if not opt then

    local opts = paths.dofile('options.lua')
    opt = opts.parse(arg)

    print('Saving everything to: ' .. opt.save)
    os.execute('mkdir -p ' .. opt.save)

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
    nEpochs = opt.nEpochs
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
            nEpochs = math.min(v[2], opt.nEpochs)
        end
        
    else
        optimStateFn = function(epoch) return optimState end
    end

    -- Random number seed
    if opt.manualSeed ~= -1 then torch.manualSeed(opt.manualSeed)
    else torch.seed() end                           

    -- Save options to experiment directory
    torch.save(opt.save .. '/options.t7', opt)

  end

  -- FRCNN specific options
  opt.frcnn_nClasses = #dataset.data.train.classLabel
  opt.frcnn_backgroundID = opt.frcnn_nClasses + 1
  opt.frcnn_num_fg_samples_per_image = math.floor(opt.frcnn_rois_per_img * opt.frcnn_fg_fraction +0.5)
  opt.frcnn_num_bg_samples_per_image = opt.frcnn_rois_per_img - opt.frcnn_num_fg_samples_per_image


  -------------------------------------------------------------------------------
  -- Preprocess rois
  -------------------------------------------------------------------------------

  local rois_preprocessed
  -- check if cache data exists already, if not do roi data preprocess 
  if paths.filep(paths.concat(opt.expDir, 'rois_cache.t7')) then
      rois_preprocessed = torch.load(paths.concat(opt.expDir, 'rois_cache.t7'))
  else
      local PreprocessROIsFn = paths.dofile('rois.lua')
      rois_preprocessed = PreprocessROIsFn(dataset, rois, opt.frcnn_fg_thresh, opt.verbose)
      torch.save(paths.concat(opt.expDir, 'rois_cache.t7'), rois_preprocessed)
  end
  opt.nClasses = #dataset.data.train.classLabel

  -- set means, stds for the regressor layer normalization
  --local roi_means = torch.cat(rois_preprocessed.train.means:view(-1,1), torch.zeros(4,1), 1)
  --local roi_stds = torch.cat(rois_preprocessed.train.stds:view(-1,1), torch.ones(4,1), 1)

  --if opt.verbose then print('Compute bbox regression mean/std values...') end
  --self.bbox_regr = self:setupData()
  --if opt.verbose then print('Done') end


  -------------------------------------------------------------------------------
  -- Setup criterion
  -------------------------------------------------------------------------------

  paths.dofile('BBoxRegressionCriterion.lua')
  local criterion
  if opt.has_bbox_regressor then
      criterion = nn.ParallelCriterion()
          :add(nn.CrossEntropyCriterion(), 1)
          --:add(nn.WeightedSmoothL1Criterion(), 1)
          :add(nn.BBoxRegressionCriterion(), 1)
  else
      criterion = nn.CrossEntropyCriterion()
  end
  local modelOut = nn.Sequential()


  -------------------------------------------------------------------------------
  -- Setup model
  -------------------------------------------------------------------------------

  if opt.GPU >= 1 then
      print('Running on GPU: num_gpus = [' .. opt.nGPU .. ']')
      require 'cutorch'
      require 'cunn'
      opt.data_type = 'torch.CudaTensor'
      model:cuda()
      criterion:cuda()
    
      -- require cudnn if available
      if pcall(require, 'cudnn') then
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
      model:float()
      criterion:float()
  end
  
  local function cast(x) return x:type(opt.data_type) end
  
  -- add mean/std norm 
  --[[
  modelOut:add(model)
  modelOut:add(nn.ParallelTable()
      :add(nn.Identity())
      :add(nn.BBoxNorm(rois_preprocessed.train.meanstd.mean, rois_preprocessed.train.meanstd.std)))
  --]]
  
  --
  model:add(nn.ParallelTable()
      :add(nn.Identity())
      :add(nn.BBoxNorm(rois_preprocessed.train.meanstd.mean, rois_preprocessed.train.meanstd.std)))    
    modelOut:add(model)
  --
  cast(modelOut)
  
  --[[
  -- normalize regressor
  if model.regressor then
      local regressor = model.regressor
      roi_means = cast(roi_means)
      roi_stds = cast(roi_stds)
      regressor.weight = regressor.weight:cdiv(roi_stds:expandAs(regressor.weight))
      regressor.bias = regressor.bias - roi_means:view(-1)
      regressor.bias = regressor.bias:cdiv(roi_stds:view(-1))
  end
  --]]
  
  if opt.verbose then
      print('Network:')
      print(model)
  end
  
  return opt, rois_preprocessed, modelOut, criterion, optimStateFn, nEpochs, rois_preprocessed.train.meanstd
end

---------------------------------------------------------------------------------------------------------------------

return LoadConfigs