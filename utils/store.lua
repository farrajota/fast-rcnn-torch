--[[
    Logger functions for easier logging data management.
]]


local utils = paths.dofile('model_utils.lua')

------------------------------------------------------------------------------------------------------------

local function store(model, modelParameters, optimState, epoch, opt, flag)
    local filename_model, filename_optimstate
    local info = 'This file contains the trained fast-rcnn network and its transformation parameters (pixel scale, colourspace, mean/std).'
   
    if flag then
        filename_model = paths.concat(opt.savedir,'model_' .. epoch ..'.t7')
        filename_optimstate = paths.concat(opt.savedir,'optim_' .. epoch ..'.t7')
    else
        filename_model = paths.concat(opt.savedir,'model.t7')
        filename_optimstate = paths.concat(opt.savedir,'optim.t7')
    end
   
    print('Saving model snapshot to: ' .. filename_model)
    torch.save(filename_optimstate, optimState)
    torch.save(filename_model, {model:clearState(), modelParameters, info})
   
    -- make a symlink to the last trained model
    local filename_symlink = paths.concat(opt.savedir,'model_final.t7')
    if paths.filep(filename_symlink) then
        os.execute(('rm %s'):format(filename_symlink))
    end
    os.execute(('ln -s %s %s'):format(filename_model, filename_symlink))
end

------------------------------------------------------------------------------------------------------------  

local function storeModel(model, modelParameters, optimState, epoch, maxepoch, opt)
    -- store model snapshot
    if opt.snapshot > 0 then
        if epoch%opt.snapshot == 0 or epoch == maxepoch then 
            store(model, modelParameters, optimState, epoch, opt, true)
        end
    elseif opt.snapshot < 0 then
        if epoch%math.abs(opt.snapshot) == 0 or epoch == maxepoch then 
            store(model, modelParameters, optimState, epoch, opt, false)
        end
    else 
        -- save only at the last epoch
        if epoch == maxepoch then
            store(model, modelParameters, optimState, epoch, opt, false)
        end
    end
end

------------------------------------------------------------------------------------------------------------

return storeModel