--[[
    Model tester class. Tests pascal voc/coco mAP of a model on a given dataset + roi proposals.
]]


local tds = require 'tds'
local xlua = require 'xlua'
--local eval = require 'fastrcnn.eval'
--local utils = require 'fastrcnn.utils'
local eval = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/eval/init.lua')
local utils = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/utils/init.lua')

if not fastrcnn then fastrcnn = {} end

------------------------------------------------------------------------------------------------

local Tester = torch.class('fastrcnn.Tester')

function Tester:__init(dataLoadTable, roi_proposals, model, modelParameters, opt, eval_mode, annotation_file)

    assert(dataLoadTable)
    assert(roi_proposals)
    assert(model)
    assert(modelParameters)
    assert(opt)

    -- initializations
    self.eval_mode = eval_mode or 'voc'
    self.progressbar = opt.progressbar or false
    self.annFile = annotation_file
    --opt.model_params = modelParameters

    self.dataLoadFn = dataLoadTable.test
    assert(self.dataLoadFn)
    self.nFiles = self.dataLoadFn.nfiles
    self.classes = self.dataLoadFn.classLabel
    self.nClasses = #self.classes

    if roi_proposals.test then
        self.roi_proposals = roi_proposals.test
    else
        self.roi_proposals = roi_proposals
    end

    -- thresholds
    self.thresh = torch.ones(self.nClasses):mul(-1.5)
    self.test_nms_thresh = opt.frcnn_test_nms_thresh
    self.test_bbox_voting_nms_thresh = opt.test_bbox_voting_nms_thresh or 0.5

    self.test_bbox_voting = false

    -- convert batchnorm from cudnn to nn (cudnn has a limit of 1024 roi boxes per batch)
    utils.model.ConvertBNcudnn2nn(model)

    -- Set image detector object
    self.ImageDetector = fastrcnn.ImageDetector(model, modelParameters, opt) -- single image detector/tester


    -- set model to test mode
    model:evaluate()


    self.cache_filename = paths.concat(opt.savedir, 'cache_tester.t7')
end

------------------------------------------------------------------------------------------------------------

function Tester:getImage(idx)
    return image.load(self.dataLoadFn.getFilename(idx),3,'float')
end

------------------------------------------------------------------------------------------------------------

function Tester:getProposals(idx)
    return self.roi_proposals[idx]
end

------------------------------------------------------------------------------------------------------------

function Tester:testOne(ifile)
    local dataset = self.dataset
    local thresh = self.thresh

    local img_boxes = tds.hash()

    -- init timers
    local timer = torch.Timer()
    local timer2 = torch.Timer()
    local timer3 = torch.Timer()

    timer:reset()

    -- load image + boxes
    local im = self:getImage(ifile)
    local boxes = self:getProposals(ifile)

    timer3:reset()

    -- check if proposal boxes exist
    local output, bbox_pred
    local tt2, nms_time
    if boxes:numel()>0 then

        local all_output = {}
        local all_bbox_pred = {}

        -- detect image
        output, bbox_pred = self.ImageDetector:detect(im, boxes)

        -- clamp predictions within image
        local bbox_pred_tmp = bbox_pred:view(-1, 2)
        bbox_pred_tmp:select(2,1):clamp(1, im:size(3))
        bbox_pred_tmp:select(2,2):clamp(1, im:size(2))

        table.insert(all_output, output)
        table.insert(all_bbox_pred, bbox_pred)

        output = utils.table.joinTable(all_output, 1)
        bbox_pred = utils.table.joinTable(all_bbox_pred, 1)

        tt2 = timer3:time().real

        timer2:reset()
        local nms_timer = torch.Timer()
        for j = 1, self.nClasses do
            local scores = output:select(2, j+1)
            local idx = torch.range(1, scores:numel()):long()
            local idx2 = scores:gt(thresh[j])
            idx = idx[idx2]
            local scored_boxes = torch.FloatTensor(idx:numel(), 5)
            if scored_boxes:numel() > 0 then
                local bx = scored_boxes:narrow(2, 1, 4)
                bx:copy(bbox_pred:narrow(2, j*4+1, 4):index(1, idx))
                scored_boxes:select(2, 5):copy(scores[idx2])
            end

            -- apply non-maxmimum suppression
            img_boxes[j] = utils.nms.fast(scored_boxes, self.test_nms_thresh)

            if self.test_bbox_voting then
                local rescaled_scored_boxes = scored_boxes:clone()
                local scores = rescaled_scored_boxes:select(2,5)
                scores:pow(opt.test_bbox_voting_score_pow or 1)

                img_boxes[j] = utils.nms.bbox_vote(img_boxes[j], rescaled_scored_boxes, self.test_bbox_voting_nms_thresh)
            end
        end
        nms_time = nms_timer:time().real

    else
        img_boxes = torch.FloatTensor()
        output = torch.FloatTensor()
        bbox_pred = torch.FloatTensor()
    end

    if ifile%1==0 and not self.progressbar then
        print(('test: %5d/%-5d dev: %d, forward time: %.3f, '
        .. 'select time: %.3fs, nms time: %.3fs, '
        .. 'total time: %.3fs'):format(ifile, self.nFiles,
        cutorch.getDevice(),
        tt2, timer2:time().real,
        nms_time, timer:time().real));
    end

    return img_boxes, {output, bbox_pred}
end

------------------------------------------------------------------------------------------------------------

function Tester:test()

    -- table initializations
    local aboxes_t = tds.hash()
    local raw_output = tds.hash()
    local raw_bbox_pred = tds.hash()

    local aboxes
    --self.cache_filename = '/home/mf/Toolkits/Codigo/git/fastrcnn-example/data/cache.t7'
    --self.cache_filename = '/home/mf/Toolkits/Codigo/git/pedestrian_detector/data/exp/caltech/alexnet_vanilla_frcnn/cache.t7'
    --if paths.filep(self.cache_filename) then
    --   print('Load test cache from file: ' .. self.cache_filename)
    --   aboxes = torch.load(self.cache_filename)
    --else



    if self.progressbar then xlua.progress(0, self.nFiles) end
    for ifile = 1, self.nFiles do
        local img_boxes, raw_boxes = self:testOne(ifile)
        aboxes_t[ifile] = img_boxes

        -- progress bar
        if self.progressbar then xlua.progress(ifile, self.nFiles) end
    end


    aboxes_t = self:keepTopKPerImage(aboxes_t, 100) -- coco only accepts 100/image
    aboxes = self:transposeBoxes(aboxes_t)
    aboxes_t = nil

    collectgarbage()

    --print('Save test cache to file: ' .. self.cache_filename)
    --torch.save(self.cache_filename, aboxes)
    --end

    return self:computeAP(aboxes)
end

------------------------------------------------------------------------------------------------------------

function Tester:keepTopKPerImage(aboxes_t, k)
    for j = 1,self.nFiles do
        aboxes_t[j] = utils.box.keep_top_k(aboxes_t[j], k)
    end
    return aboxes_t
end

------------------------------------------------------------------------------------------------------------

function Tester:transposeBoxes(aboxes_t)
    local aboxes = tds.hash()
    for j = 1, self.nClasses do
        aboxes[j] = tds.hash()
        for i = 1, self.nFiles do
            aboxes[j][i] = aboxes_t[i][j]
        end
    end
    return aboxes
end

------------------------------------------------------------------------------------------------------------

function Tester:computeAP(aboxes)
    if self.eval_mode == 'voc' then
        print('\n***************************************************')
        print('***   Pascal VOC evaluation metric   **************')
        print('***************************************************\n')
        eval.pascal(self.dataLoadFn, aboxes)
    else
        assert(self.annFile, 'Annotation file missing. Must input a valid file in order to use the coco evaluation.')
        print('\n*********************************************')
        print('***   COCO evaluation metric   **************')
        print('*********************************************\n')
        eval.coco(self.dataLoadFn, aboxes, self.annFile)
    end
end

