--[[
    Logger functions for easier logging data management.
]]


local utils = paths.dofile('utils.lua')

------------------------------------------------------------------------------------------------------------

local function regressorUndoNormalization(model, means, stds)
  if model.regressor then
      if means then
          local regressor = model.regressor
          regressor.weight = regressor.weight:cmul(stds:expand(regressor.weight:size()))
          regressor.bias = regressor.bias:cmul(stds:view(-1)) + means:view(-1)
      end
  end
  collectgarbage()
end

------------------------------------------------------------------------------------------------------------

local function regressorRedoNormalization(model, means, stds)
  if model.regressor then
      if means then
          local regressor = model.regressor
          regressor.weight = regressor.weight:cdiv(stds:expand(regressor.weight:size()))
          regressor.bias = regressor.bias - means:view(-1)
          regressor.bias = regressor.bias:cdiv(stds:view(-1))
      end
  end
  collectgarbage()
end

------------------------------------------------------------------------------------------------------------

local function store(model, optimState, epoch, opt, flag)
   local flag_optimize = flag_optimize or false
   local filename
   
   if flag then
      filename = paths.concat(opt.save,'model_' .. epoch ..'.t7')
      print('Saving model snapshot to: ' .. filename)
      utils.saveDataParallel(filename, model:clearState(), opt.GPU)
      
      torch.save(paths.concat(opt.save,'optim_' .. epoch ..'.t7'), optimState)
      
   else
      filename = paths.concat(opt.save,'model.t7')
      print('Saving model snapshot to: ' .. filename)
      utils.saveDataParallel(filename, model:clearState(), opt.GPU)
      
      torch.save(paths.concat(opt.save,'optim.t7'), optimState)
      
   end
   
   -- make a symlink to the last trained model
   local filename_symlink = paths.concat(opt.save,'model_final.t7')
   os.execute(('rm %s'):format(filename_symlink))
   os.execute(('ln -s %s %s'):format(filename, filename_symlink))
   
end

------------------------------------------------------------------------------------------------------------  

local function storeModel(model, optimState, epoch, means, stds, opt)
  
   -- undo box regressor weights normalization
   regressorUndoNormalization(model, means, stds)
   
   -- store model snapshot
   if opt.snapshot > 0 then
      if epoch%opt.snapshot == 0 then 
         store(model, optimState, epoch, opt, true)
      end
   
   elseif opt.snapshot < 0 then
      if epoch%math.abs(opt.snapshot) == 0 then 
         store(model, optimState, epoch, opt, false)
      end
   else 
      -- save only at the last epoch
      if epoch == opt.nEpochs then
         store(model, optimState, epoch, opt, false)
      end
   end
   
   -- redo box regressor weights normalization
   regressorRedoNormalization(model, means, stds)
   
end

return storeModel