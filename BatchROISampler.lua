--[[
    Samples batches of data for train/test.

    Some parts of this file's code' has been derived from https://github.com/facebookresearch/multipathnet.
]]


require 'xlua'
local utils = require 'fastrcnn.utils'

if not fastrcnn then fastrcnn = {} end

---------------------------------------------------------------------------------------------

local BatchSampler = torch.class('fastrcnn.BatchROISampler')

function BatchSampler:__init(dataLoadFn, proposals, modelParameters, opt, mode)
    assert(dataLoadFn)
    assert(proposals)
    assert(modelParameters)
    assert(opt)
    assert(mode)

    self.dataset = fastrcnn.ROIProcessor(dataLoadFn, proposals, opt)

    self.batch_size = opt.frcnn_rois_per_img or 128
    self.fg_fraction = opt.frcnn_fg_fraction or 0.25
    self.bg_fraction = opt.frcnn_bg_fraction or 1

    self.fg_num_each = self.fg_fraction * self.batch_size
    self.bg_num_each = self.batch_size - self.fg_num_each

    self.fg_threshold = opt.frcnn_fg_thresh or 0.5
    self.bg_threshold_hi = opt.frcnn_bg_thresh_hi or 0.5
    self.bg_threshold_lo = opt.frcnn_bg_thresh_lo or 0.1

    self.imgs_per_batch = opt.frcnn_imgs_per_batch or 2
    self.scale = (mode=='train' and opt.frcnn_scales) or (mode=='test' and opt.frcnn_test_scales) or 600
    self.max_size = (mode=='train' and opt.frcnn_max_size) or (mode=='test' and opt.frcnn_test_max_size) or 1000
    self.data_transformer = fastrcnn.Transform(modelParameters, opt, mode)

    self.verbose = opt.verbose or false
    self.bbox_meanstd = opt.bbox_meanstd
    self.nFiles = self.dataset.nFiles
end

------------------------------------------------------------------------------------------------------------

-- Prepare foreground / background rois for one image
-- there is a check if self.bboxes has a table prepared for this image already
-- because we prepare the rois during training to save time on loading
function BatchSampler:setupOne(idx)
    local rec = self.dataset:getProposals(idx)
    if not rec then
        return nil
    end

    local fg = rec.overlap:ge(self.fg_threshold):nonzero()
    local bg = rec.overlap:ge(self.bg_threshold_lo):cmul(rec.overlap:lt(self.bg_threshold_hi)):nonzero()
    local bg_no_overlap = rec.overlap:lt(math.max(self.bg_threshold_lo, 1e-8)):nonzero()
    return {
       [-1] = self:takeSubset(rec, bg_no_overlap, idx, true),
       [0] = self:takeSubset(rec, bg, idx, true),
       [1] = self:takeSubset(rec, fg, idx, false)
    }
end

------------------------------------------------------------------------------------------------------------

function BatchSampler:takeSubset(rec, t, idx, is_bg)
    local ind = torch.type(t) == 'table' and torch.LongTensor(t) or t:long()
    local n = ind:numel()
    if n == 0 then return end
    if ind:dim() == 2 then ind = ind:select(2,1) end
    local window = {
        indexes = torch.IntTensor(n),
        rois = torch.FloatTensor(n,4),
        labels = torch.IntTensor(n):fill(1),
        gtboxes = torch.FloatTensor(n,4):zero(),
        size = function() return n end,
    }
    window.indexes:fill(idx)
    window.rois:copy(rec.boxes:index(1,ind))
    if not is_bg then
        window.labels:add(rec.label:index(1,ind))
        local corresp = rec.correspondance:index(1,ind)
        window.gtboxes:copy(rec.boxes:index(1, corresp))
    end
    return window
end

------------------------------------------------------------------------------------------------------------

-- Calculate rois and supporting data for 'nSamples' images
-- to compute mean/var for bbox regresion
function BatchSampler:setupData(nSamples)
    local regression_values = {}
    local size = nSamples or 1000
    for i=1, size do
        xlua.progress(i, size)
        local data = {}
        while not next(data) do
            local idx = torch.random(1, self.dataset.nFiles)
            local sample = self:setupOne(idx)
            if sample then
                table.insert(data, sample[1])
            end
        end
        local v = data[1]
        if v then
            table.insert(regression_values, utils.box.convertTo(v.rois, v.gtboxes))
        end
    end
    regression_values = torch.FloatTensor():cat(regression_values,1)

    self.bbox_meanstd = {
        mean = regression_values:mean(1),
        std = regression_values:std(1)
    }

    return self.bbox_meanstd
end

------------------------------------------------------------------------------------------------------------

function BatchSampler:getImage(idx)
    local im = image.load(self.dataset:getFilename(idx),3,'float')
    -- transform image
    local im_transf, im_scale, im_size, is_flipped = self.data_transformer:image(im)
    return im_transf, im_scale, im_size, is_flipped
end

------------------------------------------------------------------------------------------------------------

function BatchSampler:selectBBoxesOne(bboxes, num_max, im_scale, im_size, do_flip)
    local rois = {}
    local labels = {}
    local gtboxes = {}

    if not bboxes then return end

    local n = bboxes:size()

    local function preprocess_bbox(input, flip)
        if input:sum()==0 then return input end
        dd = input:clone():add(-1):mul(im_scale):add(1)
        if flip then
            local tt = dd[1]
            dd[1] = im_size[2]-dd[3] +1
            dd[3] = im_size[2]-tt    +1
        end
        return dd
    end

    for i=1,math.min(num_max, n) do
        local position = torch.random(n)
        table.insert(rois,    preprocess_bbox(bboxes.rois[position], do_flip):totable())
        table.insert(gtboxes, preprocess_bbox(bboxes.gtboxes[position], do_flip):totable())
        table.insert(labels,  bboxes.labels[position])
    end

    return {
        gtboxes = torch.FloatTensor(gtboxes),
        rois = torch.FloatTensor(rois),
        labels = torch.IntTensor(labels),
    }
end

------------------------------------------------------------------------------------------------------------

function BatchSampler:selectBBoxes(boxes, im_scale, im_size, do_flip)
    local bg = self:selectBBoxesOne(boxes[0], self.bg_num_each, im_scale, im_size, do_flip)
    local fg = self:selectBBoxesOne(boxes[1], self.fg_num_each, im_scale, im_size, do_flip)
    local bg_no_overlap = self:selectBBoxesOne(boxes[-1], self.bg_num_each, im_scale, im_size, do_flip)
    local bg_rois, bg_gtboxes, bg_labels
    if self.bg_fraction == 1 or bg_no_overlap == nil then
        bg_rois = bg.rois
        bg_gtboxes = bg.gtboxes
        bg_labels = bg.labels
    else
        local num_bg_samples = math.ceil(self.bg_num_each*self.bg_fraction)
        local num_oe_bg_samples = self.bg_num_each - num_bg_samples
        if num_bg_samples>0 and num_oe_bg_samples>0 then
            local bg_idx = torch.range(1,math.min(bg.labels:size(1), num_bg_samples)):long()
            local bg_no_overlap_idx = torch.range(1,math.min(bg_no_overlap.labels:size(1), num_oe_bg_samples)):long()
            bg_rois = torch.cat(bg.rois:index(1,bg_idx), bg_no_overlap.rois:index(1,bg_no_overlap_idx),1)
            bg_gtboxes = torch.cat(bg.gtboxes:index(1,bg_idx),bg_no_overlap.gtboxes:index(1,bg_no_overlap_idx),1)
            bg_labels = torch.cat(bg.labels:index(1,bg_idx),bg_no_overlap.labels:index(1,bg_no_overlap_idx),1)
        elseif num_bg_samples>0 then
            bg_rois = bg.rois
            bg_gtboxes = bg.gtboxes
            bg_labels = bg.labels
        elseif num_oe_bg_samples>0 then
            bg_rois = bg_no_overlap.rois
            bg_gtboxes = bg_no_overlap.gtboxes
            bg_labels = bg_no_overlap.labels
        else
            error('bg samples are 0')
        end
    end
    local rois = torch.FloatTensor():cat(bg_rois, fg.rois, 1)
    local gtboxes = torch.FloatTensor():cat(bg_gtboxes, fg.gtboxes, 1)
    local labels = torch.IntTensor():cat(bg_labels, fg.labels, 1)
    return rois, labels, gtboxes
end

------------------------------------------------------------------------------------------------------------

function BatchSampler:getSample(idx)
    -- fetch boxes
    local boxes = self:setupOne(idx)
    if not boxes then
        return {}
    elseif boxes[0] == nil or boxes[1] == nil then
        return {}
    end

    -- get image
    local images, im_scale, im_size, is_flipped = self:getImage(idx)

    -- get rois, labels and ground-truth boxes
    local rois, labels, gtboxes = self:selectBBoxes(boxes, im_scale, im_size, is_flipped)

    -- get bbox regression values
    local bboxregr_vals = torch.FloatTensor(rois:size(1), 4*(#self.dataset.classes+1)):zero()

    for i,label in ipairs(labels:totable()) do
        if label > 1 then
            local out = bboxregr_vals[i]:narrow(1,(label-1)*4 + 1,4)
            utils.box.convertTo(out, rois[i], gtboxes[i])
            out:add(-1,self.bbox_meanstd.mean):cdiv(self.bbox_meanstd.std)
        end
    end

    return {images, rois, labels, bboxregr_vals}
end

------------------------------------------------------------------------------------------------------------

local max_w, max_h = 0, 0  -- this will automatically grow as the image sizes increase (up to a limit ofc)

function BatchSampler:getBatch()

    -- Load data samples
    local batchData, data, imUsed = {}, {}, {}
    for i=1, self.imgs_per_batch do
        local data = {}
        while not next(data) do
            local idx = torch.random(1, self.nFiles)
            if not imUsed[idx] then
                --print('idx: ' .. idx)
                data = self:getSample(idx)
                imUsed[idx] = 1
            end
        end
        table.insert(batchData, data)
    end

    -- image
    for i=1, #batchData do
        max_w = math.min(math.max(max_w, batchData[i][1]:size(3)), self.max_size)
        max_h = math.min(math.max(max_h, batchData[i][1]:size(2)), self.max_size)
    end
    local img = torch.FloatTensor(self.imgs_per_batch,3, max_h, max_w):fill(0)
    for i=1, self.imgs_per_batch do
        local im = batchData[i][1]
        img[{i, {}, {1,im:size(2)}, {1,im:size(3)}}]:copy(im)
    end
    --

    -- concatenate
    local boxes, labels, bbox_targets
    for i=1, self.imgs_per_batch do
        if boxes then
            boxes = boxes:cat(torch.FloatTensor(batchData[i][2]:size(1)):fill(i):cat(batchData[i][2],2),1)
            labels = labels:cat(batchData[i][3],1)
            bbox_targets = bbox_targets:cat(batchData[i][4],1)
        else
            boxes = torch.FloatTensor(batchData[i][2]:size(1)):fill(i):cat(batchData[i][2],2)
            labels = batchData[i][3]
            bbox_targets = batchData[i][4]
        end
    end

    -- randomize indexes
    local random_ind = torch.randperm(labels:size(1)):long()
    boxes = boxes:index(1, random_ind)
    labels = labels:index(1, random_ind)
    bbox_targets = bbox_targets:index(1, random_ind)


    collectgarbage()

    return {{img, boxes}, {labels, {labels, bbox_targets}}}
end
