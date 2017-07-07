--[[
    Fast-RCNN Torch7 options.
]]


if not fastrcnn then fastrcnn = {} end

------------------------------------------------------------------------------------------------------------

local Options = torch.class('fastrcnn.Options')

function Options:__init(opts)
    self.options = opts
end

------------------------------------------------------------------------------------------------------------

function Options:parse(opts)
    local options = opts or self.options
    assert(options, 'Must insert options.')

    local opt = {}

    _, opt.savedir, opt.manualSeed, opt.GPU, opt.nGPU,
    opt.nThreads, opt.verbose, opt.progressbar, opt.printConfusion,
    opt.LR, opt.LRdecay, opt.momentum, opt.weightDecay, opt.optMethod,
    opt.threshold, opt.trainIters, opt.epochStart, opt.schedule, opt.continue,
    opt.clear_buffers, opt.snapshot, opt.optimize, opt.testInter, opt.grad_clip, opt.frcnn_scales,
    opt.frcnn_max_size, opt.frcnn_imgs_per_batch, opt.frcnn_rois_per_img,
    opt.frcnn_fg_fraction, opt.frcnn_bg_fraction, opt.frcnn_fg_thresh,
    opt.frcnn_bg_thresh_hi, opt.frcnn_bg_thresh_lo, opt.frcnn_bbox_thresh,
    opt.frcnn_test_scales, opt.frcnn_test_max_size, opt.frcnn_test_max_boxes_split,
    opt.frcnn_test_nms_thresh, opt.frcnn_test_bbox_voting_nms_thresh, opt.frcnn_test_mode,
    opt.frcnn_hflip, opt.frcnn_roi_augment_offset = xlua.unpack(
    {options},
    'Parameters',
    'Fast-RCNN options',

    -------------------------------------------------------------------------------
    -- General options
    -------------------------------------------------------------------------------
    {arg='savedir', type='string', default='./data/exp/default',
     help='Store all files in the specified path.'},
    {arg='manualSeed', type='number', default=2,
     help='Manually set RNG seed.'},
    {arg='GPU', type='number', default=1,
     help='Default preferred GPU, if set to -1: no GPU.'},
    {arg='nGPU', type='number', default=1,
     help='Number of GPUs to use by default.'},
    {arg='nThreads', type='number', default=4,
     help='Number of data loading threads.'},
    {arg='verbose', type='boolean', default=true,
     help='Output messages on screen.'},
    {arg='progressbar', type='boolean', default=true,
     help='Display batch messages using a progress bar if true, else display a more verbose text info.'},
    {arg='printConfusion', type='boolean', default=false,
     help='Print confusion matrix into the screen.'},

    -------------------------------------------------------------------------------
    -- Hyperparameter options
    -------------------------------------------------------------------------------
    {arg='LR', type='number', default=1e-3,
     help='Learning rate.'},
    {arg='LRdecay', type='number', default=0.0,
     help='Learning rate decay.'},
    {arg='momentum', type='number', default=0.9,
     help='Momentum.'},
    {arg='weightDecay', type='number', default=5e-4,
     help='Weight decay.'},
    {arg='optMethod', type='string', default='sgd',
     help='ptimization method: rmsprop | sgd | nag | adadelta | adam.'},
    {arg='threshold', type='number', default=0.001,
     help='Threshold (on validation accuracy growth) to cut off training early.'},

    -------------------------------------------------------------------------------
    -- Trainer options
    -------------------------------------------------------------------------------
    {arg='trainIters', type='number', default=1000,
     help='Number of train iterations per epoch.'},
    {arg='epochStart', type='number', default=1,
     help='Manual epoch number (useful on restarts).'},
    {arg='schedule', type='table', default={{30,1e-3,5e-4},{10,1e-4,5e-4}},
     help='Optimization schedule. Overrides the previous configs if not empty.'},
    {arg='continue', type='boolean', default=false,
     help='Pick up where an experiment left off'},
    {arg='clear_buffers', type='boolean', default=false,
     help='Empty network\'s buffers (gradInput, etc) before saving the network to disk (if true)'},
    {arg='snapshot', type='number', default=10,
     help='How often to take a snapshot of the model (0 = never).'},
    {arg='optimize', type='boolean', default=true,
     help='Optimize network memory usage using optnet.'},
    {arg='testInter', type='boolean', default=true,
     help='If true, does intermediate testing of the model. Else it only tests the network at the end of the train.'},
    {arg='grad_clip', type='number', default=0,
     help='Gradient clipping (to prevent exploding gradients).'},

    -------------------------------------------------------------------------------
    -- FRCNN Training options
    -------------------------------------------------------------------------------
    {arg='frcnn_scales', type='number', default=600,
     help='Image scales -- the short edge of input image.'},
    {arg='frcnn_max_size', type='number', default=1000,
     help='Max pixel size of a scaled input image.'},
    {arg='frcnn_imgs_per_batch', type='table', default=2,
     help='Images per batch.'},
    {arg='frcnn_rois_per_img', type='number', default=128,
     help='mini-batch size (1 = pure stochastic).'},
    {arg='frcnn_fg_fraction', type='number', default=0.25,
     help='raction of minibatch that is foreground labeled (class > 0).'},
    {arg='frcnn_bg_fraction', type='number', default=1.00,
     help='Fraction of background samples that has overlap with objects (overlap >= bg_thresh_lo).'},
    {arg='frcnn_fg_thresh', type='number', default=0.5,
     help='Overlap threshold for a ROI to be considered foreground (if >= fg_thresh).'},
    {arg='frcnn_bg_thresh_hi', type='number', default=0.5,
     help='Overlap threshold for a ROI to be considered background (class = 0 if overlap in [frcnn_bg_thresh_lo, frcnn_bg_thresh_hi)).'},
    {arg='frcnn_bg_thresh_lo', type='number', default=0.1,
     help='Overlap threshold for a ROI to be considered background (class = 0 if overlap in [frcnn_bg_thresh_lo, frcnn_bg_thresh_hi)).'},
    {arg='frcnn_bbox_thresh', type='number', default=0.5,
     help='Valid training sample (IoU > bbox_thresh) for bounding box regresion.'},

    -------------------------------------------------------------------------------
    -- FRCNN Test options
    -------------------------------------------------------------------------------
    {arg='frcnn_test_scales', type='number', default=600,
     help='Image scales -- the short edge of input image.'},
    {arg='frcnn_test_max_size', type='number', default=1000,
     help='Max pixel size of a scaled input image.'},
    {arg='frcnn_test_max_boxes_split', type='number', default=2000,
     help='Split boxes proposals into segments of maximum size \'N\' (helps in out-of-memory situations)'},
    {arg='frcnn_test_nms_thresh', type='number', default=0.3,
     help='Non-Maximum suppression threshold.'},
    {arg='frcnn_test_bbox_voting_nms_thresh', type='number', default=0.5,
     help='BBox voting Non-Maximum suppression threshold.'},
    {arg='frcnn_test_mode', type='string', default='voc',
     help='mAP testing format: voc, coco'},

    -------------------------------------------------------------------------------
    -- FRCNN data augment options
    -------------------------------------------------------------------------------
    {arg='frcnn_hflip', type='number', default=0.5,
     help='Probability to flip the image horizontally [0,1].'},
    {arg='frcnn_roi_augment_offset', type='number', default=0,
     help='Increase the number of region proposals used for train between a range of coordinates defined by this value [0,1].'}
    )

    if opt.GPU >= 1 then
        opt.dataType = 'torch.CudaTensor'
    else
        opt.dataType = 'torch.FloatTensor'
    end

    self.opts = opt

    return opt
end
