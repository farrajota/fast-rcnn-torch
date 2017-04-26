--[[
    Pascal VOC mAP evaluation.
]]


--local box = require 'fastrcnn.utils.box'
--local boxoverlap = box.boxoverlap
local utils = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/utils/init.lua')
local boxoverlap = utils.box.boxoverlap


---------------------------------------------------------------------------------------------------------------------

-- source: https://github.com/fmassa/object-detection.torch/blob/master/utils.lua#L61
local function VOCap(rec,prec)
--[[ compute average precision ]]
    local ap = 0
    for t=0,1,0.1 do
        local c = prec[rec:ge(t)]
        local p
        if c:numel() > 0 then
            p = torch.max(c)
        else
            p = 0
        end
        ap=ap+p/11
    end
    return ap
end

---------------------------------------------------------------------------------------------------------------------

-- adapted from: https://github.com/fmassa/object-detection.torch/blob/master/utils.lua#L128
local function VOCevaldet(BBoxLoaderFn, nFiles, scored_boxes, classID)
--[[ compute average precision (AP), recall and precision ]]

    assert(BBoxLoaderFn)
    assert(nFiles)
    assert(scored_boxes)
    assert(classID)

    local num_pr = 0
    local energy = {}
    local correct = {}

    local count = 0

    for ifile=1, nFiles do
        -- fetch all bboxes belonging to this file and for this classID
        local bbox = {}
        local det = {}

        local boxes, labels = BBoxLoaderFn(ifile)
        
        if boxes ~= nil and labels ~=nil then
            for ibb=1, boxes:size(1) do
                if labels[ibb] == classID then
                    table.insert(bbox, boxes[ibb]:totable())
                    table.insert(det, 0)
                    count = count + 1
                end
            end
        end

        bbox = torch.Tensor(bbox)
        det = torch.Tensor(det)

        local num = scored_boxes[ifile]:numel()>0 and scored_boxes[ifile]:size(1) or 0
        for j=1, num do
            local bbox_pred = scored_boxes[ifile][j]
            num_pr = num_pr + 1
            table.insert(energy, bbox_pred[5])

            if bbox:numel()>0 then
                local o = boxoverlap(bbox,bbox_pred[{{1,4}}])
                local maxo,index = o:max(1)
                maxo = maxo[1]
                index = index[1]
                if maxo >=0.5 and det[index] == 0 then
                    correct[num_pr] = 1
                    det[index] = 1
                else
                    correct[num_pr] = 0
                end
            else
                correct[num_pr] = 0
            end
        end
    end

    if #energy == 0 then
        return 0, torch.Tensor(), torch.Tensor()
    end

    energy = torch.Tensor(energy)
    correct = torch.Tensor(correct)

    local threshold,index = energy:sort(true)

    correct = correct:index(1,index)

    -- compute recall + precision
    local n = threshold:numel()

    local recall = torch.zeros(n)
    local precision = torch.zeros(n)

    local num_correct = 0

    for i = 1,n do
        --compute precision
        local num_positive = i
        num_correct = num_correct + correct[i]
        if num_positive ~= 0 then
            precision[i] = num_correct / num_positive;
        else
            precision[i] = 0;
        end

        --compute recall
        recall[i] = num_correct / count
    end

    -- compute average precision
    local ap = VOCap(recall, precision)

    -- outputs
    return ap, recall, precision
end

---------------------------------------------------------------------------------------------------------------------

local function evaluate(BBoxLoaderFn, nfiles, classes, aboxes)
    assert(BBoxLoaderFn)
    assert(nfiles)
    assert(classes)
    assert(aboxes)

     -- (2) Compute mAP of the selected boxes wrt the ground truth boxes from the dataset
    print('==> Computing mean average precision')
    print('==> [class name] | [average precision]')
    local res = {}
    for iclass=1, #classes do
        local className = classes[iclass]
        res[iclass] = VOCevaldet(BBoxLoaderFn, nfiles, aboxes[iclass], iclass)
        print(('%s AP: %0.5f'):format(className, res[iclass]))
    end
    res = torch.Tensor(res)
    local mAP = res:mean()
    print('\n*****************')
    print(('mean AP: %0.5f'):format(mAP))
    print('*****************\n')
end

----------------------------------------------------------------------------------------

return evaluate