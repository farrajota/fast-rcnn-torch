--[[
    Process ROI samples.
]]


local utils = require 'fastrcnn.utils'
local boxoverlap = utils.box.boxoverlap

if not fastrcnn then fastrcnn = {} end

---------------------------------------------------------------------------------------------------

local ROIProcessor = torch.class('fastrcnn.ROIProcessor')

function ROIProcessor:__init(dataLoadFn, proposals, opt)
    assert(dataLoadFn, 'Invalid input: dataLoadFn')
    assert(proposals, 'Invalid input: proposals')
    assert(opt, 'Invalid input: opt')

    self.dataLoadFn = dataLoadFn
    self.roidb = proposals
    self.classes = dataLoadFn.classLabel
    self.nFiles = dataLoadFn.nfiles
    self.augment_offset = opt.frcnn_roi_augment_offset
end

------------------------------------------------------------------------------------------------------------

function ROIProcessor:getROIBoxes(idx)
    return self.roidb[idx]:float()
end

------------------------------------------------------------------------------------------------------------

function ROIProcessor:getGTBoxes(idx)
    return self.dataLoadFn.getGTBoxes(idx)
end

------------------------------------------------------------------------------------------------------------

function ROIProcessor:getFilename(idx)
    return self.dataLoadFn.getFilename(idx)
end

------------------------------------------------------------------------------------------------------------

function ROIProcessor:augmentRoiProposals(boxes)
    -- augment number of region proposals by using coordinate offset
    local new_boxes = {}
    if self.augment_offset > 0 then
        --print('Augmenting the number of roi proposal regions by jittering the available rois coordinates...')
        --local tic = torch.tic()
        local offset = torch.range(0.1, self.augment_offset, 0.1):totable()
        offset = offset[#offset]
        local roi_data = boxes:clone()
        for ix=-offset, offset, 0.1 do
        for iy=-offset, offset, 0.1 do
            if not (ix == iy and math.abs(ix) <= 0.0001) then
                local roi_data_offset = boxes:clone()

                local offx = (roi_data_offset:select(2,3) - roi_data_offset:select(2,1)):mul(ix)
                local offy = (roi_data_offset:select(2,4) - roi_data_offset:select(2,2)):mul(iy)

                roi_data_offset:select(2,1):add(offx)
                roi_data_offset:select(2,2):add(offy)
                roi_data_offset:select(2,3):add(offx)
                roi_data_offset:select(2,4):add(offy)
                roi_data = roi_data:cat(roi_data_offset, 1)
            end
        end
      end
      --roi_data[roi_data:lt(1)] = 1
      new_boxes = roi_data
        --print('Done. Elapsed time: ' .. torch.toc(tic))
    else
        new_boxes = boxes
    end

    return new_boxes
end

------------------------------------------------------------------------------------------------------------

function ROIProcessor:getProposals(idx)

    -- fetch object boxes, classes
    local gt_boxes, gt_classes = self.dataLoadFn.getGTBoxes(idx)

    -- check if there are any roi boxes for the current image
    if gt_boxes == nil then
        return nil
    end

    -- fetch roi proposal boxes
    local boxes = self:getROIBoxes(idx)

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

    -- augment the number of roi proposals
    all_boxes = self:augmentRoiProposals(all_boxes)

    -- box overlap
    rec.overlap_class = torch.FloatTensor(all_boxes:size(1), #self.classes):fill(0)
    rec.overlap = torch.FloatTensor(all_boxes:size(1), num_gt_boxes):fill(0)
    for idx=1,num_gt_boxes do
        local o = boxoverlap(all_boxes, gt_boxes[idx])
        local tmp = rec.overlap_class[{{}, gt_classes[idx]}] -- pointer copy
        tmp[tmp:lt(o)] = o[tmp:lt(o)]
        rec.overlap[{{}, idx}] = boxoverlap(all_boxes, gt_boxes[idx])
    end

    -- correspondence
    if num_gt_boxes > 0 then
        rec.overlap, rec.correspondance = rec.overlap:max(2)
        rec.overlap = torch.squeeze(rec.overlap,2)
        rec.correspondance  = torch.squeeze(rec.correspondance,2)
        rec.correspondance[rec.overlap:eq(0)] = 0
    else
        --rec.overlap = torch.FloatTensor(num_boxes+num_gt_boxes):fill(0)
        --rec.correspondance = torch.LongTensor(num_boxes+num_gt_boxes):fill(0)
        rec.overlap = torch.FloatTensor(all_boxes:size(1)):fill(0)
        rec.correspondance = torch.LongTensor(all_boxes:size(1)):fill(0)
    end

    -- set class label
    --rec.label = torch.IntTensor(num_boxes+num_gt_boxes):fill(0)
    --for idx=1,(num_boxes+num_gt_boxes) do
    rec.label = torch.IntTensor(all_boxes:size(1)):fill(0)
    for idx=1, all_boxes:size(1) do
        local corr = rec.correspondance[idx]
        if corr > 0 then
            rec.label[idx] = gt_classes[corr]
        end
    end

    rec.boxes = all_boxes
    if num_gt_boxes > 0 and num_boxes > 0 then
        rec.class = torch.cat(torch.CharTensor(gt_classes), torch.CharTensor(all_boxes:size(1)-1):fill(0))
    elseif num_boxes > 0 then
        rec.class = torch.CharTensor(all_boxes:size(1)-1):fill(0)
    elseif num_gt_boxes > 0 then
        rec.class = torch.CharTensor(gt_classes)
    else
        rec.class = torch.CharTensor(0)
    end

    function rec:size()
       return all_boxes:size(1)
    end

    return rec
end