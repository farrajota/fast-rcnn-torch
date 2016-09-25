--[[
    Load model into memory.
]]


--------------------------------------------------------------------------------
-- Features load functions 
--------------------------------------------------------------------------------

-- Note: Only the following networks are supported by these loading functions: alexnet, zeiler, vgg, resnet and googlenet v3.

local function loadAlexnetFeatures(pathModel)
  assert(pathModel)
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  require 'inn'
  
  -- load model
  local model = torch.load(pathModel)
  
  -- fetch features only
  local features = model:clone()
  features:remove(features:size()) 
  features:remove(features:size())
  features:remove(features:size())
  features:remove(features:size()) 
  features:remove(features:size()) 
  features:remove(features:size())
  features:remove(features:size())
  
  -- freeze the first layer (+ save gpu memory/computations when backprop)
  features.modules[1].accGradParameters = function() end
  features.modules[1].parameters = function() return nil end
  
  -- return features
  return features
end

----------------------------------

local function loadZeilerFeatures(pathModel)
  assert(pathModel)
  require 'cutorch'
  require 'cunn'
  
  -- load model
  local model = torch.load(pathModel)
  
  -- fetch features only
  local features = model.modules[1]:clone()
  
  -- freeze the first layer (+ save gpu memory/computations when backprop)
  features.modules[1].accGradParameters = function() end
  features.modules[1].parameters = function() return nil end
  
  -- return features
  return features
end

----------------------------------

local function loadVGGFeatures(pathModel)
  assert(pathModel)
  require 'cutorch'
  require 'cunn'
  require 'cudnn'

  -- load model
  local features = torch.load(pathModel)
  features:remove(features:size()) -- remove logsoftmax layer
  features:remove(features:size()) -- remove 3rd linear layer
  features:remove(features:size()) -- remove 2nd dropout layer
  features:remove(features:size()) -- remove 2nd last relu layer
  features:remove(features:size()) -- remove 2nd linear layer
  features:remove(features:size()) -- remove 1st dropout layer
  features:remove(features:size()) -- remove 1st relu layer
  features:remove(features:size()) -- remove 1st linear layer
  features:remove(features:size()) -- remove view layer
  features:remove(features:size()) -- remove max pool
  
  -- freeze the first 4 layers (+ save gpu memory/computations when backprop)
  -- Freeze conv1_1
  features.modules[1].accGradParameters = function() end 
  features.modules[1].parameters = function() return nil end
  -- Freeze conv1_2
  features.modules[3].accGradParameters = function() end 
  features.modules[3].parameters = function() return nil end
  -- Freeze conv2_1
  features.modules[6].accGradParameters = function() end 
  features.modules[6].parameters = function() return nil end
  -- Freeze conv2_2
  features.modules[8].accGradParameters = function() end
  features.modules[8].parameters = function() return nil end
  
  -- return features
  return features
end

----------------------------------

local function loadResnetFeatures(pathModel)
  assert(pathModel)
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  
  -- load model
  local features = torch.load(pathModel)
  features:remove(features:size())
  features:remove(features:size())
  features:remove(features:size())
  
  -- freeze the first layer (+ save gpu memory/computations when backprop)
  -- Freeze conv1_1
  features.modules[1].accGradParameters = function() end
  features.modules[1].parameters = function() return nil end
  
  -- return features
  return features
end

----------------------------------

local function loadGooglenetFeatures(pathModel)
  assert(pathModel)
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  
  -- load model
  local features = torch.load(pathModel)
  features:remove(features:size())
  features:remove(features:size())
  features:remove(features:size())
  features:remove(features:size())
  
  -- freeze the first layer (+ save gpu memory/computations when backprop)
  -- Freeze conv1_1
  features.modules[1].accGradParameters = function() end 
  features.modules[1].parameters = function() return nil end
  
  -- return features
  return features
end

----------------------------------

local function LoadFeatures(modelName, path)

  -- verbose flag
  local verbose = verbose or false

  local model_list = { 
      alexnet     = 'alexnet.t7' ,
      zeiler      = 'zeilernet.t7',
      vgg16       = 'vgg16.t7',
      vgg19       = 'vgg19.t7',
      resnet18    = 'resnet-18.t7',
      resnet32    = 'resnet-32.t7',
      resnet50    = 'resnet-50.t7',
      resnet101   = 'resnet-101.t7',
      resnet152   = 'resnet-152.t7',
      resnet200   = 'resnet-200.t7',
      googlenetv3 = 'googlenet_inceptionv3_cudnn.t7'
  }
  
  assert(model_list[string.lower(modelName)], 'Model not defined for loading: ' .. modelName .. '. Please select one of the following available models for load: alexnet | vgg16 | vgg19 | resnet-18 | resnet-34 | resnet-50 | resnet-101 | resnet-152 | resnet-200 | zeiler | googlenetv3.' )
  
  local model_name = model_list[string.lower(modelName)]
  local filename = paths.concat(path, 'model_' .. model_name)
  if paths.filep(filename) then
      local parameters = torch.load(paths.concat(path, 'parameters_' .. model_name))
      if string.match(modelName, 'alexnet') then
        return loadAlexnetFeatures(filename), parameters
      elseif string.match(modelName, 'zeiler') then
        return loadZeilerFeatures(filename), parameters
      elseif string.match(modelName, 'vgg') then
        return loadVGGFeatures(filename), parameters
      elseif string.match(modelName, 'resnet') then
        return loadResnetFeatures(filename), parameters
      elseif string.match(modelName, 'googlenet') then
        return loadGooglenetFeatures(filename), parameters
      else
        error('Model doesn\'t exist: ' .. modelName)
      end
  else
      error('File not found: ' .. filename)
  end

end


--------------------------------------------------------------------------------
-- Setup model creation functions
--------------------------------------------------------------------------------

local function CreateROIPooler(roipool_width, roipool_height, feat_stride)
-- w: feature grids width of the detection region
-- h: feature grids height of the detection region
-- stride: number of pixels that one grid pixel in the last convolutional layer represents on the image  
  assert(roipool_width)
  assert(roipool_height)
  assert(feat_stride)
  
  return nn.ROIPooling(roipool_width,roipool_height):setSpatialScale(1/feat_stride)
end

------------------------------------------------------------------------------------------------------

local function CreateClassifier(roipool_width, roipool_height, num_feats_last_conv, nClasses, flag_train_bbox_regressor)

  assert(roipool_width)
  assert(roipool_height)
  assert(num_feats_last_conv)
  assert(nClasses)
  assert(flag_train_bbox_regressor)

  local fully_connected_layers = nn.Sequential()
  fully_connected_layers:add(nn.View(-1):setNumInputDims(3))
  fully_connected_layers:add(nn.Linear(num_feats_last_conv * roipool_width * roipool_height, 4096))
  fully_connected_layers:add(nn.BatchNormalization(4096))
  fully_connected_layers:add(nn.ReLU(true))
  fully_connected_layers:add(nn.Dropout(0.5))
  fully_connected_layers:add(nn.Linear(4096, 4096))
  fully_connected_layers:add(nn.BatchNormalization(4096))
  fully_connected_layers:add(nn.ReLU(true))
  fully_connected_layers:add(nn.Dropout(0.5))
  
  local cls_reg_layer 
  if flag_train_bbox_regressor then
      local cls = nn.Linear(4096,nClasses+1)      --classifier
      local reg = nn.Linear(4096,(nClasses+1)*4)  --regressor
      cls_reg_layer =  nn.ConcatTable():add(cls):add(reg)
  else
      cls_reg_layer = nn.Linear(4096,nClasses+1)
  end
  
  return fully_connected_layers, cls_reg_layer
end

------------------------------------------------------------------------------------------------------

local function CreateCriterion(train_bbox_regressor)
  local criterion
  print('==> Criterion')
  if train_bbox_regressor then
      criterion = nn.ParallelCriterion():add(nn.CrossEntropyCriterion(), 1):add(nn.WeightedSmoothL1Criterion(), 1)
  else
      criterion = nn.CrossEntropyCriterion()
  end
  return criterion
end

------------------------------------------------------------------------------------------------------
 
local function CreateModel(featuresNet, parameters, nClasses, opt)
  -- parameters should have the following fields: num_feats and stride.
  assert(featuresNet)
  assert(parameters)
  assert(nClasses)
  assert(opt)
  
  -- verbose flag
  local verbose = opt.verbose or false
  
  local roipool_width, roipool_height = opt.roipoolSize[1], opt.roipoolSize[2]
  
  -- (1) load roi pooling layer
  local roipooler = CreateROIPooler(roipool_width, roipool_height, parameters.stride)
  
  -- (2) Create classifier network
  local fully_connected_layers, cls_reg_layers = CreateClassifier(roipool_width, roipool_height, parameters.num_feats, nClasses, opt.train_bbox_regressor)
  
  -- (3) group parts into a single model
  local model = nn.Sequential()
      :add(nn.ParallelTable():add(featuresNet):add(nn.Identity()))
      :add(roipooler)
      :add(nn.Sequential()
          :add(fully_connected_layers)
          :add(cls_reg_layers))
  
  -- define a quick and easy lookup field for the regressor module
  if torch.type(cls_reg_layers) == 'nn.ConcatTable' then
      model.regressor = cls_reg_layers.modules[2]
  end
  
  if verbose then
      print('Features network:')
      if featuresNet:size() > 30 then
          print('<network too big to print>')
      else
          print(featuresNet)
      end
      print('Roi pooling:')
      print(roipooler)
      print('Fully-connected network:')
      print(fully_connected_layers)
      print('Classifier network:')
      print(cls_reg_layers)
  end
  
  return model, criterion
end

------------------------------------------------------------------------------------------------------

return {
    CreateModel = CreateModel,
    CreateCriterion = CreateCriterion,
    LoadFeatures = LoadFeatures
}