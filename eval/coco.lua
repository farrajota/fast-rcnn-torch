--[[
    Execute coco evaluation code in python via Lua.
]]


local function coco_eval_python(annFile, res)
    assert(annFile)
    assert(res)

    local command = ('import sys;' ..
                    'from pycocotools.coco import COCO;' ..
                    'from pycocotools.cocoeval import COCOeval;' ..
                    'cocoGt = COCO(\'%s\');' ..
                    'cocoDt = cocoGt.loadRes(\'%s\');' ..
                    'imgIds = sorted(cocoDt.imgToAnns.keys());' ..
                    'imgIds = imgIds[0:len(imgIds)];' ..
                    'cocoEval = COCOeval(cocoGt,cocoDt);' ..
                    'cocoEval.params.imgIds = imgIds;' ..
                    'cocoEval.evaluate();' ..
                    'cocoEval.accumulate();' ..
                    'cocoEval.summarize();' ..
                    'stats = cocoEval.stats;'
                    :format(annFile, res)

    os.execute(('python -c "%s"'):format(command))
end

return coco_eval_python