--[[
    Usefull utility functions for managing networks.
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

local function makeDataParallelTable(model, nGPU)
   if nGPU > 1 then
      local gpus = torch.range(1, nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            require 'nngraph'
            paths.dofile('../ROIPooling.lua')
            if pcall(require,'cudnn') then
               local cudnn = require 'cudnn'
               cudnn.fastest, cudnn.benchmark = fastest, benchmark
            end
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end
   return model
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

local function logical2ind(logical)
  if logical:numel() == 0 then
    return torch.LongTensor()
  end
  return torch.range(1,logical:numel())[logical:gt(0)]:long()
end

------------------------------------------------------------------------------------------------------------

local function tds_to_table(input)
  assert(input)
  
  ---------------------------------------------
  local function convert_tds_to_table(input)
    assert(input)
    local out = {}
    for k, v in pairs(input) do
        out[k] = v
    end
    return out
  end
  ---------------------------------------------
  
  if type(input) == 'table' then
      return input
  elseif type(input) == 'cdata' then
      if string.lower(torch.type(input)) == 'tds.hash' or string.lower(torch.type(input)) == 'tds.hash' then
          return convert_tds_to_table(input)
      else
          error('Input must be either a tds.hash, tds.vec or a table: ' .. torch.type(input))
      end
  else
      error('Input must be either a tds.hash, tds.vec or a table: ' .. torch.type(input))
  end
end

------------------------------------------------------------------------------------------------------------

return {
    -- model initializations
    MSRinit = MSRinit,
    FCinit = FCinit,
    DisableBias = DisableBias,
    
    -- parallelize networks
    makeDataParallelTable = makeDataParallelTable,
    saveDataParallel = saveDataParallel,
    loadDataParallel = loadDataParallel,
    
    -- non-maximum suppression
    nms = paths.dofile('nms.lua'),
    
    -- bounding box transformations
    box_transform = paths.dofile('bbox_transform.lua'),
    
    -- bounding box overlap
    boxoverlap = paths.dofile('boxoverlap.lua'),
    
    -- other functions
    logical2ind = logical2ind,
    
    -- VOC eval functions
    voc_eval = paths.dofile('voc_eval.lua'),
    
    -- convert a tds.hash/tds.vec into a table
    tds_to_table = tds_to_table,
}