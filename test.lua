--[[
    Test script. Computes the mAP of all proposals. Acts as a wrapper to the fastrcnn.Tester class.
--]]


local function test(dataset, roi_proposals, model, modelParameters, opt, eval_mode)

    assert(dataset)
    assert(roi_proposals)
    assert(model)
    assert(modelParameters)
    assert(opt)

    local evaluation_mode = eval_mode or 'voc'

    -- load roi boxes from file into memory
    local roi_boxes
    if roi_proposals.test then
        roi_boxes = roi_proposals.test
    else
        roi_boxes = roi_proposals
    end

    -- test class
    local Tester = fastrcnn.Tester(dataset, roi_boxes, model, modelParameters, opt, evaluation_mode)

    -- compute the mAP score
    local mAP_score = Tester:test()
    return mAP_score
end

return test