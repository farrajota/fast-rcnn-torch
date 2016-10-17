--[[
    Model utility functions.
]]


local ffi=require 'ffi'

------------------------------------------------------------------------------------------------------------

local function MSRinit(model)
   for k,v in pairs(model:findModules('nn.SpatialConvolution')) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      if v.bias then v.bias:zero() end
   end
end

------------------------------------------------------------------------------------------------------------

local function FCinit(model)
   for k,v in pairs(model:findModules'nn.Linear') do
     v.bias:zero()
   end
end

------------------------------------------------------------------------------------------------------------

local function DisableBias(model)
   for i,v in ipairs(model:findModules'nn.SpatialConvolution') do
      v.bias = nil
      v.gradBias = nil
   end
end

------------------------------------------------------------------------------------------------------------

local function makeDataParallel(module, nGPU)
  local nGPU = nGPU or 1
  if nGPU > 1 then
    local dpt = nn.DataParallelTable(1) -- true?
    local cur_dev = cutorch.getDevice()
    for i = 1, nGPU do
      cutorch.setDevice(i)
      dpt:add(module:clone():cuda(), i)
    end
    cutorch.setDevice(cur_dev)
    return dpt
  else
    return nn.Sequential():add(module)
  end
end


------------------------------------------------------------------------------------------------------------

local function cleanDPT(module, GPU)
   -- This assumes this DPT was created by the function above: all the
   -- module.modules are clones of the same network on different GPUs
   -- hence we only need to keep one when saving the model to the disk.
   local newDPT = nn.DataParallelTable(1, true, true)
   cutorch.setDevice(GPU)
   newDPT:add(module:get(1), GPU)
   return newDPT
end

------------------------------------------------------------------------------------------------------------

local function saveDataParallel(filename, model, GPU)
   if torch.type(model) == 'nn.DataParallelTable' then
      torch.save(filename, cleanDPT(model, GPU))
   elseif torch.type(model) == 'nn.Sequential' or torch.type(model) == 'nn.gModule' then
      local temp_model = nn.Sequential()
      for i, module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            temp_model:add(cleanDPT(module, GPU))
         else
            temp_model:add(module)
         end
      end
      torch.save(filename, temp_model)
   else
      error('This saving function only works with Sequential or DataParallelTable modules.')
   end
end

------------------------------------------------------------------------------------------------------------

local function loadDataParallel(filename, nGPU)
   --if opt.backend == 'cudnn' then
   --   require 'cudnn'
   --end
   local model = torch.load(filename)
   if torch.type(model) == 'nn.DataParallelTable' then
      return makeDataParallelTable(model:get(1):float(), nGPU)
   elseif torch.type(model) == 'nn.Sequential' or torch.type(model) == 'nn.gModule' then
      for i,module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            model.modules[i] = makeDataParallelTable(module:get(1):float(), nGPU)
         end
      end
      return model
   else
      error('The loaded model is not a Sequential or DataParallelTable module.')
   end
end

------------------------------------------------------------------------------------------------------------

local function ConvertBNcudnn2nn(net)
  local function ConvertModule(net)
    return net:replace(function(x)
        if torch.type(x) == 'cudnn.BatchNormalization' then
          return cudnn.convert(x, nn)
        else
          return x
        end
    end)
  end
  net:apply(function(x) return ConvertModule(x) end)
end

------------------------------------------------------------------------------------------------------------

local function DisableFeatureBackprop(features, maxLayer)
  local noBackpropModules = nn.Sequential()
  for i = 1,maxLayer do
    noBackpropModules:add(features.modules[1])
    features:remove(1)
  end
  features:insert(nn.NoBackprop(noBackpropModules), 1)
end

------------------------------------------------------------------------------------------------------------

local function CreateClassifierBBoxRegressor(nHidden, nClasses, has_bbox_regressor)
  
  assert(nHidden)
  assert(nClasses)
  
  local classifier = nn.Linear(nHidden,nClasses+1)      --classifier
  classifier.weight:normal(0,0.001)
  classifier.bias:zero()
  
  local regressor = nn.Linear(nHidden,(nClasses+1)*4)  --regressor
  regressor.weight:normal(0,0.001)
  regressor.bias:zero()
  
  if has_bbox_regressor then
      return nn.ConcatTable():add(classifier):add(regressor)
  else
      return classifier
  end
end

------------------------------------------------------------------------------------------------------------

local function AddBBoxNorm(meanstd)
  return nn.ParallelTable()
      :add(nn.Identity())
      :add(nn.BBoxNorm(meanstd.mean, meanstd.std))
end

------------------------------------------------------------------------------------------------------------

local function NormalizeBBoxRegr(model, meanstd)
  if #model:findModules('nn.BBoxNorm') == 0 then
    -- normalize the bbox regression
    local regression_layer = model:get(#model.modules):get(2)
    if torch.type(regression_layer) == 'nn.Sequential' then
      regression_layer = regression_layer:get(#regression_layer.modules)
    end
    assert(torch.type(regression_layer) == 'nn.Linear')

    local mean_hat = torch.repeatTensor(meanstd.mean,1,opt.num_classes):cuda()
    local sigma_hat = torch.repeatTensor(meanstd.std,1,opt.num_classes):cuda()

    regression_layer.weight:cdiv(sigma_hat:t():expandAs(regression_layer.weight))
    regression_layer.bias:add(-mean_hat):cdiv(sigma_hat)

    return AddBBoxNorm(model, meanstd)
  end
end

------------------------------------------------------------------------------------------------------------

return {
    MSRinit = MSRinit,
    FCinit = FCinit,
    DisableBias = DisableBias,
    
    -- parallelize networks
    makeDataParallel = makeDataParallel,
    saveDataParallel = saveDataParallel,
    loadDataParallel = loadDataParallel,
    
    ConvertBNcudnn2nn = ConvertBNcudnn2nn,
    DisableFeatureBackprop = DisableFeatureBackprop,
    CreateClassifierBBoxRegressor = CreateClassifierBBoxRegressor,
    AddBBoxNorm = AddBBoxNorm,
    NormalizeBBoxRegr = NormalizeBBoxRegr
}