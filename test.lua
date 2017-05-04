--[[
    Test script. Computes the mAP of all proposals.
    Acts as a wrapper to the fastrcnn.Tester class.
--]]


local function test(dataLoadTable, rois, model, modelParameters, opt, annotation_file)

    assert(dataLoadTable)
    assert(rois)
    assert(model)
    assert(modelParameters)
    assert(opt)

    local evaluation_mode = opt.frcnn_test_mode or 'voc'

    -- load roi boxes from file into memory
    local roi_boxes
    if rois.test then
        roi_boxes = rois.test
    else
        roi_boxes = rois
    end

    -- add model params to opt
    if not opt.model_param then
        opt.model_param = modelParameters
    end

    -- test class
    local Tester = fastrcnn.Tester(dataLoadTable(), roi_boxes, model, modelParameters, opt, evaluation_mode, annotation_file)

    -- compute the mAP score
    Tester:test()
end

return test