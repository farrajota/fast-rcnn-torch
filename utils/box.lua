--[[
    Bounding box utility functions.
]]


-- source: https://github.com/fmassa/object-detection.torch/blob/master/utils.lua#L97
local function boxoverlap(a,b)
  local b = b.xmin and {b.xmin,b.ymin,b.xmax,b.ymax} or b

  local x1 = a:select(2,1):clone()
  x1[x1:lt(b[1])] = b[1]
  local y1 = a:select(2,2):clone()
  y1[y1:lt(b[2])] = b[2]
  local x2 = a:select(2,3):clone()
  x2[x2:gt(b[3])] = b[3]
  local y2 = a:select(2,4):clone()
  y2[y2:gt(b[4])] = b[4]

  local w = x2-x1+1;
  local h = y2-y1+1;
  local inter = torch.cmul(w,h):float()
  local aarea = torch.cmul((a:select(2,3)-a:select(2,1)+1) ,
                           (a:select(2,4)-a:select(2,2)+1)):float()
  local barea = (b[3]-b[1]+1) * (b[4]-b[2]+1);

  -- intersection over union overlap
  local o = torch.cdiv(inter , (aarea+barea-inter))
  -- set invalid entries to 0 overlap
  o[w:lt(0)] = 0
  o[h:lt(0)] = 0

  return o
end

------------------------------------------------------------------------------------------------------------

-- source: https://github.com/fmassa/object-detection.torch/blob/master/utils.lua#L34
local function keep_top_k(boxes,top_k)
    local X = joinTable(boxes,1)
    if X:numel() == 0 then
        return nil
    end
    local scores = X[{{},-1}]:sort(1,true)
    local thresh = scores[math.min(scores:numel(),top_k)]
    for i=1,#boxes do
        local bbox = boxes[i]
        if bbox:numel() > 0 then
            local idx = torch.range(1,bbox:size(1)):long()
            local keep = bbox[{{},-1}]:ge(thresh)
            idx = idx[keep]
            if idx:numel() > 0 then
                boxes[i] = bbox:index(1,idx)
            else
                boxes[i]:resize()
            end
        end
    end
    return boxes, thresh
end

------------------------------------------------------------------------------------------------------------

local function convertTo(out, bbox, tbox)
-- bbox, tbox: [x1,y1,x2,y2]
   if torch.type(out) == 'table' or out:nDimension() == 1 then
      local xc = (bbox[1] + bbox[3]) * 0.5
      local yc = (bbox[2] + bbox[4]) * 0.5
      local w = bbox[3] - bbox[1]
      local h = bbox[4] - bbox[2]
      local xtc = (tbox[1] + tbox[3]) * 0.5
      local ytc = (tbox[2] + tbox[4]) * 0.5
      local wt = tbox[3] - tbox[1]
      local ht = tbox[4] - tbox[2]
      out[1] = (xtc - xc) / w
      out[2] = (ytc - yc) / h
      out[3] = math.log(wt / w)
      out[4] = math.log(ht / h)
   else
      local xc = (bbox[{{},1}] + bbox[{{},3}]) * 0.5
      local yc = (bbox[{{},2}] + bbox[{{},4}]) * 0.5
      local w = bbox[{{},3}] - bbox[{{},1}]
      local h = bbox[{{},4}] - bbox[{{},2}]
      local xtc = (tbox[{{},1}] + tbox[{{},3}]) * 0.5
      local ytc = (tbox[{{},2}] + tbox[{{},4}]) * 0.5
      local wt = tbox[{{},3}] - tbox[{{},1}]
      local ht = tbox[{{},4}] - tbox[{{},2}]
      out[{{},1}] = (xtc - xc):cdiv(w)
      out[{{},2}] = (ytc - yc):cdiv(h)
      out[{{},3}] = wt:cdiv(w):log()
      out[{{},4}] = ht:cdiv(h):log()
   end
end

------------------------------------------------------------------------------------------------------------

local function convertToMulti(...)
   local arg = {...}
   if #arg == 3 then
      convertTo(...)
   else
      local x = arg[1]:clone()
      convertTo(x, arg[1], arg[2])
      return x
   end
end

------------------------------------------------------------------------------------------------------------

local function convertFrom(out, bbox, y)
   if torch.type(out) == 'table' or out:nDimension() == 1 then
      local xc = (bbox[1] + bbox[3]) * 0.5
      local yc = (bbox[2] + bbox[4]) * 0.5
      local w = bbox[3] - bbox[1]
      local h = bbox[4] - bbox[2]

      local xtc = xc + y[1] * w
      local ytc = yc + y[2] * h
      local wt = w * math.exp(y[3])
      local ht = h * math.exp(y[4])

      out[1] = xtc - wt/2
      out[2] = ytc - ht/2
      out[3] = xtc + wt/2
      out[4] = ytc + ht/2
   else
      assert(bbox:size(2) == y:size(2))
      assert(bbox:size(2) == out:size(2))
      assert(bbox:size(1) == y:size(1))
      assert(bbox:size(1) == out:size(1))
      local xc = (bbox[{{},1}] + bbox[{{},3}]) * 0.5
      local yc = (bbox[{{},2}] + bbox[{{},4}]) * 0.5
      local w = bbox[{{},3}] - bbox[{{},1}]
      local h = bbox[{{},4}] - bbox[{{},2}]

      local xtc = torch.addcmul(xc, y[{{},1}], w)
      local ytc = torch.addcmul(yc, y[{{},2}], h)
      local wt = torch.exp(y[{{},3}]):cmul(w)
      local ht = torch.exp(y[{{},4}]):cmul(h)

      out[{{},1}] = xtc - wt * 0.5
      out[{{},2}] = ytc - ht * 0.5
      out[{{},3}] = xtc + wt * 0.5
      out[{{},4}] = ytc + ht * 0.5
   end
end

------------------------------------------------------------------------------------------------------------

return {
    boxoverlap = boxoverlap,

    keep_top_k = keep_top_k,

    convertTo = convertTo,
    convertToMulti = convertToMulti,
    convertFrom = convertFrom,
}