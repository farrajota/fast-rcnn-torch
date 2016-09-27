--[[
    Loads necessary libraries and files for the train script.
]]


local function LoadConfigs(model, dataset, rois)

  -------------------------------------------------------------------------------
  -- Load necessary libraries and files
  -------------------------------------------------------------------------------

  paths.dofile('WeightedSmoothL1Criterion.lua')
  local modelFn = paths.dofile('model.lua')
  local roisFn = paths.dofile('rois.lua')

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
        local schedule = opt.schedule
        optimStateFn = function(epoch) 
            for k, v in pairs(schedule) do
                if v[1] <= epoch and v[2] >= epoch then
                  return {
                      learningRate = v[3],
                      learningRateDecay = opt.LRdecay,
                      momentum = opt.momentum,
                      dampening = 0.0,
                      weightDecay = v[4]
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
  local roi_means = torch.cat(rois_preprocessed.train.means:view(-1,1), torch.zeros(4,1), 1)
  local roi_stds = torch.cat(rois_preprocessed.train.stds:view(-1,1), torch.ones(4,1), 1)


  -------------------------------------------------------------------------------
  -- Load criterion
  -------------------------------------------------------------------------------

  local criterion = modelFn.CreateCriterion(opt.train_bbox_regressor)
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
          --cudnn.convert(model, cudnn):cuda()
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

  -- normalize regressor
  if model.regressor then
      local regressor = model.regressor
      roi_means = cast(roi_means)
      roi_stds = cast(roi_stds)
      regressor.weight = regressor.weight:cdiv(roi_stds:expandAs(regressor.weight))
      regressor.bias = regressor.bias - roi_means:view(-1)
      regressor.bias = regressor.bias:cdiv(roi_stds:view(-1))
  end

  -- Use multiple gpus
  if opt.GPU >= 1 and opt.nGPU > 1 then
    local utils = paths.dofile('util/utils.lua')
    modelOut:add(model) -- copy the entire model
    modelOut.modules[1].modules[1] = utils.makeDataParallelTable(model.modules[1], opt.nGPU)-- parallelize only the features layer
    --modelOut:add( utils.makeDataParallelTable(model, opt.nGPU))
  else
    modelOut:add(model)
  end

  cast(modelOut)
  
  return opt, rois_preprocessed, modelOut, criterion, optimStateFn, nEpochs, roi_means, roi_stds
end

---------------------------------------------------------------------------------------------------------------------

return LoadConfigs