local ROIPooling,parent = torch.class('nn.ROIPooling','nn.Module')

function ROIPooling:__init(W,H)
  parent.__init(self)
  self.W = W
  self.H = H
  self.pooler = {}--nn.SpatialAdaptiveMaxPooling(W,H)
  self.spatial_scale = 1
  self.gradInput = {torch.Tensor(), torch.Tensor()}
  self._rois = torch.FloatTensor()
end

function ROIPooling:setSpatialScale(scale)
  self.spatial_scale = scale
  return self
end

function ROIPooling:updateOutput(input)
  local data = input[1]
  local rois = input[2]

  local num_rois = rois:size(1)
  local s = data:size()
  local ss = s:size(1)
  self.output:resize(num_rois,s[ss-2],self.H,self.W)

  -- element access is faster if not a cuda tensor
  if rois:type() == 'torch.CudaTensor' then
    self._rois:resize(rois:size()):copy(rois)
    rois = self._rois
  else
    rois = self._rois:resize(rois:size()):copy(rois)
  end

  rois[{{},{2,5}}]:add(-1):mul(self.spatial_scale):add(1):round()
  rois[{{},2}]:cmin(s[ss])
  rois[{{},3}]:cmin(s[ss-1])
  rois[{{},4}]:cmin(s[ss])
  rois[{{},5}]:cmin(s[ss-1])


  if not self._type then self._type = self.output:type() end

  if #self.pooler < num_rois then
    local diff = num_rois - #self.pooler
    for i=1,diff do
      table.insert(self.pooler,nn.SpatialAdaptiveMaxPooling(self.W,self.H):type(self._type))
    end
  end

  for i=1,num_rois do
    --print(i)
    local roi = rois[i]
    local im_idx = roi[1]
    local im = data[{im_idx,{},{roi[3],roi[5]},{roi[2],roi[4]}}]
    self.output[i] = self.pooler[i]:updateOutput(im)
  end
  return self.output
end

function ROIPooling:updateGradInput(input,gradOutput)
  if self.gradInput[1] == nil then
    self.gradInput = {nn.utils.recursiveType(torch.Tensor(), torch.type(input[1])), 
                      nn.utils.recursiveType(torch.Tensor(), torch.type(input[2]))} 
    -- NOTE: clearState() removes this tensor, so this is just a quick fix.
  end
  
  local data = input[1]
  local rois = input[2]
  if rois:type() == 'torch.CudaTensor' then
    rois = self._rois
  end
  local num_rois = rois:size(1)
  local s = data:size()
  local ss = s:size(1)
  self.gradInput[1]:resizeAs(data):zero()
  self.gradInput[2]:resizeAs(input[2]):zero()

  for i=1,num_rois do
    local roi = rois[i]
    local im_idx = roi[1]
    local r = {im_idx,{},{roi[3],roi[5]},{roi[2],roi[4]}}
    local im = data[r]
    local g  = self.pooler[i]:updateGradInput(im,gradOutput[i])
    self.gradInput[1][r]:add(g)
  end
  -- this clears memory for the poolers, which might cause huge memory storage issues if kept untouched
  for i=1,num_rois do
    self.pooler[i]:clearState()
  end
  
  return self.gradInput
end

function ROIPooling:type(type)
  parent.type(self,type)
  for i=1,#self.pooler do
    self.pooler[i]:type(type)
  end
  self._type = type
  return self
end

function ROIPooling:__tostring__()
  return torch.type(self) .. string.format('(%d x %d -> step: %d)', self.W, self.H, 1/self.spatial_scale)
end

