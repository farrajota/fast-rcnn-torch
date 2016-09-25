--[[
    Fast-RCNN Torch7 implementation. Options script.
]]

local options = {}

-- (2) Load options
function options.parse(arg)
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Torch-7 Fast-RCNN script.')
  cmd:text()
  cmd:text(' ---------- General options ------------------------------------')
  cmd:text()
  cmd:option('-expID',        'alexnet_t2', 'Experiment ID')
  cmd:option('-dataset', 'pascalvoc2007', 'Dataset choice: pascalvoc2007 | pascalvoc2012 | mscoco')
  cmd:option('-expDir',   projectDir .. '/data/exp',  'Experiments directory')
  cmd:option('-manualSeed',          2, 'Manually set RNG seed')
  cmd:option('-GPU',                 1, 'Default preferred GPU, if set to -1: no GPU')
  cmd:option('-nGPU',                2, 'Number of GPUs to use by default')
  cmd:option('-nThreads',            2, 'Number of data loading threads')
  cmd:option('-verbose',        "true", 'Output messages on screen.')
  cmd:text()
  cmd:text(' ---------- Model options --------------------------------------')
  cmd:text()
  cmd:option('-netType',       'alexnet', 'Options: alexnet | vgg16 | vgg19 | resnet-18 | resnet-34 | resnet-50 | ' ..                                                 'resnet-101 | resnet-152 | resnet-200 | zeiler | googlenetv3.')
  cmd:option('-loadModel',      'none', 'Provide the name of a previously trained model')
  cmd:option('-continue',      "false", 'Pick up where an experiment left off')
  cmd:option('-snapshot',            5, 'How often to take a snapshot of the model (0 = never)')
  cmd:option('-optimize',       "true", 'Optimize network memory usage using optnet.')
  cmd:option('-roipoolSize',   "{7,7}", 'RoI pooling size.')
  cmd:option('-train_bbox_regressor',  "true", 'Enable/disable model\'s box predictor.')
  cmd:option('-train_full_model',      "true", 'Train full model (classifier + fine-tune features if true) or only the classifier (if false).')
  cmd:text()
  cmd:text(' ---------- Hyperparameter options -----------------------------')
  cmd:text()
  cmd:option('-LR',               1e-3, 'Learning rate')
  cmd:option('-LRdecay',           0.0, 'Learning rate decay')
  cmd:option('-momentum',          0.9, 'Momentum')
  cmd:option('-weightDecay',       5e-4, 'Weight decay')
  cmd:option('-optMethod',   'sgd', 'Optimization method: rmsprop | sgd | nag | adadelta')
  cmd:option('-threshold',        .001, 'Threshold (on validation accuracy growth) to cut off training early')
  cmd:text()
  cmd:text(' ---------- Training options -----------------------------------')
  cmd:text()
  cmd:option('-nEpochs',           10, 'Total number of epochs to run')
  cmd:option('-epochNumber',       1, 'Manual epoch number (useful on restarts)')
  cmd:option('-schedule', {{4,1e-3,5e-4},{3, 1e-4, 5e-4}}, 'Optimization schedule. Overrides the previous configs if not empty.')
  cmd:option('-testInter',   "true", 'If true, does intermediate testing of the model. Else it only tests the network at the end of the train.')
  cmd:text()
  cmd:text()
  cmd:text(' ===============================================================')
  cmd:text(' ========== ***Fast RCNN options*** ============================')
  cmd:text(' ===============================================================')
  cmd:text()
  cmd:text(' ---------- FRCNN Train options --------------------------------------')
  cmd:text()
  cmd:option('-frcnn_scales',        "{600}", 'Image scales -- the short edge of input image.')
  cmd:option('-frcnn_max_size',       1000, 'Max pixel size of a scaled input image.')
  cmd:option('-frcnn_imgs_per_batch',    2, 'Images per batch.')
  cmd:option('-frcnn_rois_per_img',     64, 'mini-batch size (1 = pure stochastic)')
  cmd:option('-frcnn_fg_fraction',    0.25, 'Fraction of minibatch that is foreground labeled (class > 0).')
  cmd:option('-frcnn_bg_fraction',    0.75, 'Fraction of background samples that has overlap with objects (overlap >= bg_thresh_lo).')
  cmd:option('-frcnn_fg_thresh',       0.5, 'Overlap threshold for a ROI to be considered foreground (if >= fg_thresh).')
  cmd:option('-frcnn_bg_thresh_hi',    0.5, 'Overlap threshold for a ROI to be considered background (class = 0 if overlap in [frcnn_bg_thresh_lo, frcnn_bg_thresh_hi))')
  cmd:option('-frcnn_bg_thresh_lo',    0.1, 'Overlap threshold for a ROI to be considered background (class = 0 if overlap in [frcnn_bg_thresh_lo, frcnn_bg_thresh_hi)).')
  cmd:option('-frcnn_bbox_thresh',     0.5, 'Valid training sample (IoU > bbox_thresh) for bounding box regresion.')
  cmd:text()
  cmd:text(' ---------- FRCNN Test options --------------------------------------')
  cmd:text()
  cmd:option('-frcnn_test_scales',    "{600}", 'Image scales -- the short edge of input image.')
  cmd:option('-frcnn_test_max_size',   1000, 'Max pixel size of a scaled input image.')
  cmd:option('-frcnn_test_nms_thresh',  0.3, 'Non-Maximum suppression threshold.')
  cmd:option('-frcnn_test_binary',    "false", 'Network save name.')
  cmd:text()
  cmd:text(' ---------- FRCNN Addon options --------------------------------------')
  cmd:text()
  cmd:option('-frcnn_augment_percent',    0,  'Defines the pixel offset of the boxes edges to define the lower and upper boundary of the jittering window. Value must be in [0,1] range.')
  cmd:option('-frcnn_quantity',           1, 'Augments the number of roi boxes by this ammount. It is used to define the step size (num_step_size = floor(sqrt(augment_quantity)); step_size=offset*2/num_step_size) of the bbox coordinates jittering. Value must be  greater than 0.')
  cmd:text()
  cmd:text(' ---------- FRCNN data augment options --------------------------------------')
  cmd:text()
  cmd:option('-frcnn_hflip',         0.5, 'Probability to flip the image horizontally [0,1].')
  cmd:option('-frcnn_rotate',          0, 'Rotation angle.')
  cmd:option('-frcnn_jitter',          0, 'Image jitter offset')
  cmd:text()
  
  -- parse options
  local opt = cmd:parse(arg or {})
  
  ---------------------------------------------------------------------------------------------------
  local function ConvertString2Boolean(var) -- converts string to booleans
    if type(var) == 'string' then
      local str = string.lower(var):gsub("%s+", "")
      str = string.gsub(str, "%s+", "")
      if str == 'true' then
        return true
      elseif str == 'false' then
        return false
      else
        assert(false, 'Cannot convert input to boolean type: ' .. var)
      end
    elseif type(var) == 'boolean' then
      return var
    else
      assert(false, 'Input variable is not of string/boolean type: ' .. type(var))
    end
  end
  
  ---------------------------------------------------------------------------------------------------
  
  local function Str2TableFn(input) -- convert a string into a table 
      local json = require 'rapidjson'

      -- replace '{' and '}' by '[' and '], respectively
      input = input:gsub("%{","[")
      input = input:gsub("%}","]")
      
      -- use json decode function to convert the string into a table
      return json.decode(input)
  end
  ---------------------------------------------------------------------------------------------------
  
  opt.expDir = paths.concat(opt.expDir, opt.dataset)
  opt.save = paths.concat(opt.expDir, opt.expID)
  opt.load = (opt.loadModel and opt.loadModel ~= 'none') or 'model_final.t7'

  -- check if some booleans were inserted as strings. If so, convert the string to boolean type
  --opt.force_preprocess = ConvertString2Boolean(opt.force_preprocess)
  opt.train_bbox_regressor = ConvertString2Boolean(opt.train_bbox_regressor)
  opt.train_full_model = ConvertString2Boolean(opt.train_full_model)
  opt.verbose = ConvertString2Boolean(opt.verbose)
  
  opt.roipoolSize = Str2TableFn(opt.roipoolSize)
  opt.frcnn_scales = Str2TableFn(opt.frcnn_scales)
  opt.frcnn_test_scales = Str2TableFn(opt.frcnn_test_scales)
  
  if opt.GPU >= 1 then
      opt.dataType = 'torch.CudaTensor'
  else
      opt.dataType = 'torch.FloatTensor'
  end
  
 
  return opt
end

return options