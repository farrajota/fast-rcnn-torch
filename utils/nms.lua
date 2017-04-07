--[[
     Non-maximum suppression.
]]

local ffi = require 'ffi'

ffi.cdef[[
void bbox_vote(THFloatTensor *res, THFloatTensor *nms_boxes, THFloatTensor *scored_boxes, float threshold);
void NMS(THFloatTensor *keep, THFloatTensor *scored_boxes, float overlap);
]]

--local ok, C = pcall(ffi.load, package.searchpath('libfastrcnn', package.cpath))
local ok, C = pcall(ffi.load, '/home/mf/Toolkits/Codigo/git/fastrcnn/lib/libnms.so')
assert(ok, 'Installation went wrong when compiling the C code.')

------------------------------------------------------------------------------------------------------------

local function bbox_vote(nms_boxes, scored_boxes, threshold)
   local res = torch.FloatTensor()
   C.bbox_vote(res:cdata(), nms_boxes:cdata(), scored_boxes:cdata(), threshold)
   return res
end

------------------------------------------------------------------------------------------------------------

local function nms_fast(boxes, overlap)
   local keep = torch.FloatTensor()
   C.NMS(keep:cdata(), boxes:cdata(), overlap)
   return keep
end

------------------------------------------------------------------------------------------------------------

local function nms_dense(boxes, overlap)
    local n_boxes = boxes:size(1)

    if n_boxes == 0 then
        return torch.LongTensor()
    end

    -- sort scores in descending order
    assert(boxes:size(2) == 5)
    local vals, I = torch.sort(boxes:select(2,5), 1, true)

    -- sort the boxes
    local boxes_s = boxes:index(1, I):t():contiguous()

    local suppressed = torch.ByteTensor():resize(boxes_s:size(2)):zero()

    local x1 = boxes_s[1]
    local y1 = boxes_s[2]
    local x2 = boxes_s[3]
    local y2 = boxes_s[4]
    local s  = boxes_s[5]

    local area = torch.cmul((x2-x1+1), (y2-y1+1))

    local pick = torch.LongTensor(s:size(1)):zero()

    -- these clones are just for setting the size
    local xx1 = x1:clone()
    local yy1 = x1:clone()
    local xx2 = x1:clone()
    local yy2 = x1:clone()
    local w = x1:clone()
    local h = x1:clone()

    local pickIdx = 1
    for c = 1, n_boxes do
        if suppressed[c] == 0 then
            pick[pickIdx] = I[c]
            pickIdx = pickIdx + 1

            xx1:copy(x1):clamp(x1[c], math.huge)
            yy1:copy(y1):clamp(y1[c], math.huge)
            xx2:copy(x2):clamp(0, x2[c])
            yy2:copy(y2):clamp(0, y2[c])

            w:add(xx2, -1, xx1):add(1):clamp(0, math.huge)
            h:add(yy2, -1, yy1):add(1):clamp(0, math.huge)
            local inter = w
            inter:cmul(h)
            local union = xx1
            union:add(area, -1, inter):add(area[c])
            local ol = h
            torch.cdiv(ol, inter, union)

            suppressed:add(ol:gt(overlap)):clamp(0,1)
        end
    end

    pick = pick[{{1,pickIdx-1}}]
    return pick
end

------------------------------------------------------------------------------------------------------------

return {
    fast = nms_fast,
    dense = nms_dense,

    bbox_vote = bbox_vote
}