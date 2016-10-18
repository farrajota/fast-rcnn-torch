--[[
    Logger functions for easier logging data management.
]]


local utils = paths.dofile('model_utils.lua')

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
   if paths.filep(filename_symlink) then
      os.execute(('rm %s'):format(filename_symlink))
   end
   os.execute(('ln -s %s %s'):format(filename, filename_symlink))
   
end

------------------------------------------------------------------------------------------------------------  

local function storeModel(model, optimState, epoch, maxepoch, opt)
  
   -- store model snapshot
   if opt.snapshot > 0 then
      if epoch%opt.snapshot == 0 or epoch == maxepoch then 
         store(model, optimState, epoch, opt, true)
      end
   
   elseif opt.snapshot < 0 then
      if epoch%math.abs(opt.snapshot) == 0 or epoch == maxepoch then 
         store(model, optimState, epoch, opt, false)
      end
   else 
      -- save only at the last epoch
      if epoch == maxepoch then
         store(model, optimState, epoch, opt, false)
      end
   end
   
end

return storeModel