--[[
    Process ROI samples.
]]

local boxoverlap = paths.dofile('utils/boxoverlap.lua')
local ffi = require 'ffi'

local ROIProcessor =  torch.class('fastrcnn.ROIProcessor')

---------------------------------------------------------------------------------------------------

function ROIProcessor:__init(dataset, proposals, opt)
    assert(dataset)
    assert(proposals)
    assert(opt)
    
    self.dataset = dataset
    self.roidb = proposals
    self.classes = self.dataset.classLabel
    self.nFiles = dataset.filename:size(1)
end


function ROIProcessor:getROIBoxes(idx)
    return self.roidb[idx]
end

--[[
function ROIProcessor:getGTBoxes(idx)
    local object_ids = self.dataset.filenameList.objectIDList[idx]:gt(0):squeeze(1):nonzero()
    local gt_boxes, gt_classes = {}, {}
    for i=1, object_ids:size(1) do
        local objID = self.dataset.filenameList.objectIDList[idx][object_ids[i][1] ]
        local bbox = self.dataset.bbox[self.dataset.object[objID][3] ]
        local label = self.dataset.object[objID][2]
        table.insert(gt_boxes, bbox:totable())
        table.insert(gt_classes, label)
    end
    gt_boxes = torch.FloatTensor(gt_boxes)
    return gt_boxes,gt_classes
end
--]]

function ROIProcessor:getGTBoxes(idx)
    local size = self.dataset.filenameList.objectIDList[idx]:size(1)
    local gt_boxes, gt_classes = {}, {}
    for i=1, size do
        local objID = self.dataset.filenameList.objectIDList[idx][i]
        if objID == 0 then
            break
        end
        local bbox = self.dataset.bbox[self.dataset.object[objID][3]]
        local label = self.dataset.object[objID][2]
        table.insert(gt_boxes, bbox:totable())
        table.insert(gt_classes, label)
    end
    gt_boxes = torch.FloatTensor(gt_boxes)
    return gt_boxes,gt_classes
end


function ROIProcessor:getFilename(idx)
    local filename = ffi.string(self.dataset.filename[idx]:data())
    return filename
end

function ROIProcessor:getProposals(idx)
  
    -- check if there are any roi boxes for the current image
    if self.dataset.filenameList.objectIDList[idx]:sum() == 0 then
        return nil
    end
    
    -- fetch roi proposal boxes
    local boxes = self:getROIBoxes(idx)
    
    -- fetch object boxes, classes 
    local gt_boxes, gt_classes = self:getGTBoxes(idx)
    
    local all_boxes 
    if boxes:numel() > 0 and gt_boxes:numel() > 0 then
        all_boxes = torch.cat(gt_boxes,boxes,1)
    elseif boxes:numel() == 0 then
        all_boxes = gt_boxes
    else
        all_boxes = boxes
    end
    
    local num_boxes = boxes:numel() > 0 and boxes:size(1) or 0
    local num_gt_boxes = #gt_classes
    
    -- data recipient
    local rec = {}
    if num_gt_boxes > 0 and num_boxes > 0 then
        rec.gt = torch.cat(torch.ByteTensor(num_gt_boxes):fill(1), torch.ByteTensor(num_boxes):fill(0))
    elseif num_boxes > 0 then
        rec.gt = torch.ByteTensor(num_boxes):fill(0)
    elseif num_gt_boxes > 0 then
        rec.gt = torch.ByteTensor(num_gt_boxes):fill(1)
    else
        rec.gt = torch.ByteTensor(0)
    end

    -- box overlap
    rec.overlap_class = torch.FloatTensor(num_boxes+num_gt_boxes, #self.classes):fill(0)
    rec.overlap = torch.FloatTensor(num_boxes+num_gt_boxes,num_gt_boxes):fill(0)
    for idx=1,num_gt_boxes do
        local o = boxoverlap(all_boxes,gt_boxes[idx])
        local tmp = rec.overlap_class[{{},gt_classes[idx]}] -- pointer copy
        tmp[tmp:lt(o)] = o[tmp:lt(o)]
        rec.overlap[{{},idx}] = boxoverlap(all_boxes,gt_boxes[idx])
    end
    
    -- correspondence
    if num_gt_boxes > 0 then
        rec.overlap, rec.correspondance = rec.overlap:max(2)
        rec.overlap = torch.squeeze(rec.overlap,2)
        rec.correspondance  = torch.squeeze(rec.correspondance,2)
        rec.correspondance[rec.overlap:eq(0)] = 0
    else
        rec.overlap = torch.FloatTensor(num_boxes+num_gt_boxes):fill(0)
        rec.correspondance = torch.LongTensor(num_boxes+num_gt_boxes):fill(0)
    end
    
    -- set class label 
    rec.label = torch.IntTensor(num_boxes+num_gt_boxes):fill(0)
    for idx=1,(num_boxes+num_gt_boxes) do
        local corr = rec.correspondance[idx]
        if corr > 0 then
            rec.label[idx] = gt_classes[corr]
        end
    end
    
    rec.boxes = all_boxes
    if num_gt_boxes > 0 and num_boxes > 0 then
        rec.class = torch.cat(torch.CharTensor(gt_classes), torch.CharTensor(num_boxes):fill(0))
    elseif num_boxes > 0 then
        rec.class = torch.CharTensor(num_boxes):fill(0)
    elseif num_gt_boxes > 0 then
        rec.class = torch.CharTensor(gt_classes)
    else
        rec.class = torch.CharTensor(0)
    end

    function rec:size()
       return (num_boxes+num_gt_boxes)
    end

    return rec
end


return ROIProcessor