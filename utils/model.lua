--[[
    Model utility functions.
]]


require 'nn'

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

local function setDataParallel(net, GPU, nGPU)
    local function ConvertModule(net)
        return net:replace(function(x)
            if torch.type(x) == 'nn.DataParallelTable' then
                if nGPU > 1 then
                    return makeDataParallel(x:get(1), nGPU)
                else
                    return cleanDPT(x, GPU)
                end
            else
                return x
            end
        end)
    end
    net:apply(function(x) return ConvertModule(x) end)
end

------------------------------------------------------------------------------------------------------------

local function resetDataParallel(net, GPU)
    local GPU_id = GPU or 1
    if torch.type(model) == 'nn.DataParallelTable' then
        return cleanDPT(model, GPU_id)
    elseif torch.type(model) == 'nn.Sequential' or torch.type(model) == 'nn.gModule' then
        local temp_model = nn.Sequential()
        for i, module in ipairs(model.modules) do
            if torch.type(module) == 'nn.DataParallelTable' then
                temp_model:add(cleanDPT(module, GPU_id))
            else
                temp_model:add(module)
            end
        end
        return temp_model
    else
        error('This saving function only works with Sequential or DataParallelTable modules.')
    end
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
    for i=1, maxLayer do
        noBackpropModules:add(features.modules[1])
        features:remove(1)
    end
    features:insert(nn.NoBackprop(noBackpropModules), 1)
end

------------------------------------------------------------------------------------------------------------

local function CreateClassifierBBoxRegressor(nHidden, nClasses)
    assert(nHidden)
    assert(nClasses)

    local classifier = nn.Linear(nHidden,nClasses+1)      --classifier
    classifier.weight:normal(0,0.001)
    classifier.bias:zero()

    local regressor = nn.Linear(nHidden,(nClasses+1)*4)  --regressor
    regressor.weight:normal(0,0.001)
    regressor.bias:zero()

    return nn.ConcatTable():add(classifier):add(regressor)
end

------------------------------------------------------------------------------------------------------------

local function AddBBoxNorm(meanstd)
    assert(meanstd)
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

local function testSurgery(input, f, net, ...)
   local output1 = net:forward(input):clone()
   f(net,...)
   local output2 = net:forward(input):clone()
   print((output1 - output2):abs():max())
   assert((output1 - output2):abs():max() < 2e-5)
end

------------------------------------------------------------------------------------------------------------

local function snapshot_configs(model_fname, epoch, opt)
    return {
        bbox_meanstd = opt.bbox_meanstd,
        epoch = epoch,
        optimState = optimState,
        model_name = model_fname
    }
end

------------------------------------------------------------------------------------------------------------

local function store(model, modelParameters, optimState, epoch, opt, flag)
    local filename_model, filename_optimstate
    local info = 'This file contains the trained fast-rcnn network and its transformation ' ..
                 'parameters (pixel scale, colourspace, mean/std).' ..
                 '\nWarning: You must reconvert the network\'s layers back to cudnn backend if used.' ..
                 'The saving operation converts all layers to the \'nn\' backend when saving to disk.'

    if flag then
        filename_model = paths.concat(opt.savedir,'model_' .. epoch ..'.t7')
        filename_optimstate = paths.concat(opt.savedir,'optim_' .. epoch ..'.t7')
    else
        filename_model = paths.concat(opt.savedir,'model.t7')
        filename_optimstate = paths.concat(opt.savedir,'optim.t7')
    end

    print('Saving model snapshot to: ' .. filename_model)
    torch.save(filename_optimstate, optimState)
    torch.save(opt.curr_save_configs, snapshot_configs(filename_model, epoch, opt))

    if opt.clear_buffers then
        model = model:clearState()
    end
    setDataParallel(model, opt.GPU, 1)  -- set nn.DataParallelTable to use only 1 GPU
    torch.save(filename_model, {model, modelParameters, info})
    setDataParallel(model, opt.GPU, opt.nGPU)

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

return {
    MSRinit = MSRinit,
    FCinit = FCinit,
    DisableBias = DisableBias,

    -- parallelize networks
    makeDataParallel = makeDataParallel,
    setDataParallel  = setDataParallel,
    resetDataParallel = resetDataParallel,
    saveDataParallel = saveDataParallel,
    loadDataParallel = loadDataParallel,

    ConvertBNcudnn2nn = ConvertBNcudnn2nn,
    DisableFeatureBackprop = DisableFeatureBackprop,
    CreateClassifierBBoxRegressor = CreateClassifierBBoxRegressor,
    AddBBoxNorm = AddBBoxNorm,
    NormalizeBBoxRegr = NormalizeBBoxRegr,

    testSurgery = testSurgery,

    -- store model snapshot
    storeModel = storeModel
}