--[[
    Test script. Computes the mAP of all proposals.
    Acts as a wrapper to the fastrcnn.Tester class.
--]]


local function test(dataLoadTable, rois, model, modelParameters, opt)

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

    -- test class
    local Tester = fastrcnn.Tester(dataLoadTable, roi_boxes, model, modelParameters, opt, evaluation_mode)

    -- compute the mAP score
    local mAP_score = Tester:test()
    return mAP_score
end

return test