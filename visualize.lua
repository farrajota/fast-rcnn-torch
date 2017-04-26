--[[
    Visualize detections using a GUI window (requires qlua/qt).
]]

--local nms = require 'fastrcnn.utils.nms'
local nms = paths.dofile('utils/nms.lua')

---------------------------------------------------------------------------------------------------------------------

local function visualize_detections(im, boxes, scores, visualization_thresh, nms_thresh, classes)
    local ok = pcall(require, 'qt')
    if not ok then
        error('You need to run visualize_detections using qlua')
    end
    require 'qttorch'
    require 'qtwidget'

    -- clamp predictions within image
    local boxes_tmp = boxes:view(-1, 2)
    boxes_tmp:select(2,1):clamp(1, im:size(3))
    boxes_tmp:select(2,2):clamp(1, im:size(2))

    -- select best scoring boxes without background
    local max_score, maxID = scores[{{},{2,-1}}]:max(2)

    -- max id
    local idx = maxID:squeeze():gt(1):cmul(max_score:gt(visualization_thresh)):nonzero()

    if idx:numel()==0 then
        local x,y = im:size(3),im:size(2)
        local w = qtwidget.newwindow(x,y,"Fast R-CNN for Torch7! No objects detected on this frame")
        local qtimg = qt.QImage.fromTensor(im)
        w:image(0,0,x,y,qtimg)
        local fontsize = 16
        return w
    end

    idx=idx:select(2,1)
    boxes = boxes:index(1, idx)
    maxID = maxID:index(1, idx)
    max_score = max_score:index(1, idx)

    -- select bbox
    local boxes_thresh = {}
    for i=1, boxes:size(1) do
        local label = maxID[i][1]
        table.insert(boxes_thresh, boxes[i]:narrow(1,(label-1)*4 + 1,4):totable())
    end
    boxes_thresh = torch.FloatTensor(boxes_thresh)

    local scored_boxes = torch.cat(boxes_thresh:float(), max_score:float(), 2)
    local keep = nms.dense(scored_boxes, nms_thresh or 0.3)

    boxes_thresh = boxes_thresh:index(1,keep)
    max_score = max_score:index(1,keep):squeeze()
    maxID = maxID:index(1,keep):squeeze()

    local num_boxes = boxes_thresh:size(1)
    local widths  = boxes_thresh[{{},3}] - boxes_thresh[{{},1}]
    local heights = boxes_thresh[{{},4}] - boxes_thresh[{{},2}]

    local x,y = im:size(3),im:size(2)
    local w = qtwidget.newwindow(x,y,"Fast R-CNN for Torch7!")

    local qtimg = qt.QImage.fromTensor(im)
    w:image(0,0,x,y,qtimg)
    local fontsize = 16

    w:setcolor(200/255,130/255,200/255,1)
    w:rectangle(20,20,120,55)
    w:fill()
    w:stroke()

    w:setcolor(0,0,0,1)
    w:fill(false)
    w:rectangle(20,20,120,55)
    w:stroke()
    w:moveto(30,40)
    w:setfont(qt.QFont{serif=true,italic=true,size=fontsize,bold=true})
    w:setcolor(qt.QColor("#000000"))
    w:show('Click on')
    w:moveto(30,40+fontsize+5)
    w:show('boxes!')
    for i=1,num_boxes do
      local x,y = boxes_thresh[{i,1}],boxes_thresh[{i,2}]
      local width,height = widths[i], heights[i]
      w:rectangle(x,y,width,height)
    end
    w:setcolor("#7CFF00")
    w:setlinewidth(2)
    w:stroke()

    qt.connect(w.listener,
      'sigMousePress(int,int,QByteArray,QByteArray,QByteArray)',
      function(x,y)
          for i = 1, boxes_thresh:size(1) do
            if x>boxes_thresh[i][1] and x < boxes_thresh[i][3] and y>boxes_thresh[i][2] and y<boxes_thresh[i][4] then
              w:setcolor(200/255,130/255,200/255,1)
              w:rectangle(20,20,120,55)
              w:fill()
              w:stroke()

              w:setcolor(0,0,0,1)
              w:fill(false)
              w:rectangle(20,20,120,55)
              w:stroke()

              w:moveto(30,40)
              w:setfont(qt.QFont{serif=true,italic=true,size=fontsize,bold=true})
              w:setcolor(qt.QColor("#000000"))
              w:show(classes[maxID[i]])
              w:moveto(30,40+fontsize+5)
              w:show(string.format('%2.2f',max_score[i]))
              w:stroke()
              w:fill(false)
            end
          end
      end );

    return w
end

---------------------------------------------------------------------------------------------------------------------

return visualize_detections