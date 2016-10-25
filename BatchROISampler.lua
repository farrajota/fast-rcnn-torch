--[[
    Samples batches of data for train/test. 
]]

local utils = paths.dofile('utils/init.lua')

paths.dofile('ROIProcessor.lua')
paths.dofile('Transform.lua')

---------------------------------------------------------------------------------------------

local BatchSampler = torch.class('fastrcnn.BatchROISampler')


function BatchSampler:__init(dataset, proposals, opt, mode)
    assert(dataset)
    assert(proposals)
    assert(opt)
    assert(mode)
    
    self.dataset = fastrcnn.ROIProcessor(dataset, proposals, opt)
    
    self.batch_size = opt.frcnn_rois_per_img or 128
    self.fg_fraction = opt.frcnn_fg_fraction or 0.25
    
    self.fg_threshold = opt.frcnn_fg_thresh or 0.5
    self.bg_threshold_hi = opt.frcnn_bg_thresh_hi or 0.5
    self.bg_threshold_lo = opt.frcnn_bg_thresh_lo or 0.1

    self.imgs_per_batch = opt.frcnn_imgs_per_batch or 2
    self.scale = (mode=='train' and opt.frcnn_scales) or (mode=='test' and opt.frcnn_test_scales) or 600
    self.max_size = (mode=='train' and opt.frcnn_max_size) or (mode=='test' and opt.frcnn_test_max_size) or 1000
    self.data_transformer = fastrcnn.Transform(opt, mode)

    self.verbose = opt.verbose or false

    --self.scale_jitter    = scale_jitter or 0    -- uniformly jitter the scale by this frac
    --self.aspect_jitter   = aspect_jitter or 0   -- uniformly jitter the scale by this frac
    --self.crop_likelihood = crop_likelihood or 0 -- likelihood of doing a random crop (in each dimension, independently)
    --self.crop_attempts = 10                     -- number of attempts to try to find a valid crop
    --self.crop_min_frac = 0.7 -- a crop must preserve at least this fraction of the iamge
end


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
    return {
       [0] = self:takeSubset(rec, bg, idx, true),
       [1] = self:takeSubset(rec, fg, idx, false)
    }
end


function BatchSampler:takeSubset(rec, t, idx, is_bg)
    local ind = torch.type(t) == 'table' and torch.LongTensor(t) or t:long()
    local n = ind:numel()
    if n == 0 then return end
    if ind:dim() == 2 then ind = ind:select(2,1) end
    local window = {
        indexes = torch.IntTensor(n),
        rois = torch.FloatTensor(n,4),
        labels = torch.IntTensor(n):fill(#self.dataset.classes+1),
        gtboxes = torch.FloatTensor(n,4):zero(),
        size = function() return n end,
    }
    window.indexes:fill(idx)
    window.rois:copy(rec.boxes:index(1,ind))
    if not is_bg then
        window.labels:copy(rec.label:index(1,ind))
        local corresp = rec.correspondance:index(1,ind)
        window.gtboxes:copy(rec.boxes:index(1, corresp))
    end
    return window
end


-- Calculate rois and supporting data for the first 1000 images
-- to compute mean/var for bbox regresion
function BatchSampler:setupData()
    if self.verbose then print('Compute bbox regression mean/std values...') end
    
    local regression_values = {}
    local subset_size = 1000
    for i = 1,1000 do
        local v = self:setupOne(i)[1]
        if v then
            table.insert(regression_values, utils.convertTo(v.rois, v.gtboxes))
        end
    end
    regression_values = torch.FloatTensor():cat(regression_values,1)

    self.bbox_regr = {
        mean = regression_values:mean(1),
        std = regression_values:std(1)
    }
    
    if self.verbose then print('Done') end
    
    return self.bbox_regr
end


function BatchSampler:getImage(idx)
    local im = image.load(self.dataset:getFilename(idx),3,'float')
    -- transform image
    local im_transf, im_scale, im_size, is_flipped = self.data_transformer:image(im)
    return im_transf, im_scale, im_size, is_flipped
end


function BatchSampler:selectBBoxesOne(bboxes, num_max, im_scale, im_size, do_flip)
    local rois = {}
    local labels = {}
    local gtboxes = {}
    
    local n = bboxes:size()

    for i=1,math.min(num_max, n) do
        local position = torch.random(n)
        table.insert(rois,    self.data_transformer:bbox(bboxes.rois[position],im_scale, im_size, do_flip))
        table.insert(gtboxes, self.data_transformer:bbox(bboxes.gtboxes[position],im_scale, im_size, do_flip))
        table.insert(labels,  bboxes.labels[position])
    end

    return {
        gtboxes = torch.FloatTensor():cat(gtboxes,1),
        rois = torch.FloatTensor():cat(rois,1),
        labels = torch.IntTensor(labels),
    }
end


function BatchSampler:selectBBoxes(boxes, im_scale, im_size, do_flip)
    local rois = {}
    local labels = {}
    local gtboxes = {}
    for im,v in ipairs(boxes) do

        local bg = self:selectBBoxesOne(v[0], self.bg_num_each, im_scale, im_size, do_flip)
        local fg = self:selectBBoxesOne(v[1], self.fg_num_each, im_scale, im_size, do_flip)

        local imrois = torch.FloatTensor():cat(bg.rois, fg.rois, 1)
        imrois = torch.FloatTensor(imrois:size(1),1):fill(im):cat(imrois, 2)
        local imgtboxes = torch.FloatTensor():cat(bg.gtboxes, fg.gtboxes, 1)
        local imlabels = torch.IntTensor():cat(bg.labels, fg.labels, 1)

        table.insert(rois, imrois)
        table.insert(gtboxes, imgtboxes)
        table.insert(labels, imlabels)
    end
    gtboxes = torch.FloatTensor():cat(gtboxes,1)
    rois = torch.FloatTensor():cat(rois,1)
    labels = torch.IntTensor():cat(labels,1)
    return rois, labels, gtboxes
end


function BatchSampler:getSample(idx)
    -- fetch boxes
    local rec = self:setupOne(idx)
    if not rec then
        return {}
    end
    
    -- get image
    local images, im_scale, im_size, is_flipped = self:getImage(idx)
    
    -- falta sacar boxes (ver o codigo do outro)
    
    -- get rois, labels and ground-truth boxes
    local rois, labels, gtboxes = self:selectBBoxes(boxes, im_scale, im_size, is_flipped)
    
    -- get bbox regression values
    local bboxregr_vals = torch.FloatTensor(rois:size(1), 4*(#self.dataset.classes+1)):zero()
    
    for i,label in ipairs(labels:totable()) do
        if label < #self.dataset.classes+1 then
            local out = bboxregr_vals[i]:narrow(1,(label-1)*4 + 1,4)
            utils.convertTo(out, rois[i]:narrow(1,2,4), gtboxes[i])
            out:add(-1,self.bbox_regr.mean):cdiv(self.bbox_regr.std)
        end
    end

    local batches = {images, rois}
    local targets = {labels, {labels, bboxregr_vals}}

    return batches, targets
end


function BatchSampler:getBatch()
  
    -- Load data samples
    local batchData, data, imUsed = {}, {}, {}
    for i=1, self.imgs_per_batch do
        local data = {}
        while not next(data) do
            local idx = torch.random(1, self.dataset.nFiles)
            if not imUsed[idx] then 
                data = self:getSample(idx)
                imUsed[idx] = 1
            end
        end
        table.insert(batchData, data)
    end
    
    -- image
    --local img = torch.FloatTensor(num_images_per_batch,3,max_height, max_width):fill(0)
    local img = torch.FloatTensor(self.imgs_per_batch,3, self.max_size, self.max_size):fill(0)
    for i=1, self.imgs_per_batch do
        local im = data[i][1]
        img[{i, {}, {1,im:size(2)}, {1,im:size(3)}}]:copy(im)
    end

    -- concatenate
    local boxes, labels, bbox_targets, loss_weights
    for i=1, self.imgs_per_batch do
        if boxes then 
            boxes = boxes:cat(torch.FloatTensor(data[i][2]:size(1)):fill(i):cat(data[i][2],2),1)
            labels = labels:cat(data[i][3],1)
            bbox_targets = bbox_targets:cat(data[i][4],1)
        else
            boxes = torch.FloatTensor(data[i][2]:size(1)):fill(i):cat(data[i][2],2)
            labels = data[i][3]
            bbox_targets = data[i][4]
        end
    end
    
    collectgarbage()
    
    return {{img, boxes}, {labels, {labels, bbox_targets}}}
end
