--[[
    Function to visualize detections (needs qlua/qt).
]]

local nms = paths.dofile('nms.lua')

local function visualize_detections(im, boxes, scores, thresh, nms_thresh, cl_names)
  local ok = pcall(require,'qt')
  if not ok then
    error('You need to run visualize_detections using qlua')
  end
  require 'qttorch'
  require 'qtwidget'

  -- select best scoring boxes without background
  local max_score,idx = scores[{{},{1, #cl_names}}]:max(2)

  local idx_thresh = max_score:gt(thresh)
  max_score = max_score[idx_thresh]
  idx = idx[idx_thresh]

  local r = torch.range(1,boxes:size(1)):long()
  local rr = r[idx_thresh]
  if rr:numel() == 0 then
    error('No detections with a score greater than the specified threshold')
  end
  local boxes_thresh = boxes:index(1,rr)
  
  local keep = nms(torch.cat(boxes_thresh:float(),max_score:float(),2), nms_thresh)
  
  boxes_thresh = boxes_thresh:index(1,keep)
  max_score = max_score:index(1,keep)
  idx = idx:index(1,keep)

  local num_boxes = boxes_thresh:size(1)
  local widths  = boxes_thresh[{{},3}] - boxes_thresh[{{},1}]
  local heights = boxes_thresh[{{},4}] - boxes_thresh[{{},2}]

  local x,y = im:size(3),im:size(2)
  local w = qtwidget.newwindow(x,y,"Fast R-CNN for Torch!")
  
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
            w:show(cl_names[idx[i]])
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