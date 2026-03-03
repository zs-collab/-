flowchart LR
  X[LR Gray Input\nx: (B,1,H,W)] --> HEAD[Head Conv3x3\n1 -> C (n_feats)]
  X --> TEX[TextureBranchGray\nstem(Conv+GELU)x2\n+ num_tex_blocks x TextureMixBlock\n+ conv_out]
  TEX --> TEXFEAT[tex_feat: (B,tex_ch,H,W)]
  TEXFEAT --> TEXPROJ[tex_proj 1x1\ntex_ch -> C]
  TEXFEAT --> TEXGATE[tex_gate 1x1 + Sigmoid\ntex_ch -> C]
  TEXPROJ --> TEXMUL[tex * tex_gate]
  TEXGATE --> TEXMUL

  X --> FREQOPT{use_freq_branch?}
  FREQOPT -->|Yes| FREQ[FrequencyBranchGray\nFFT high-pass\n+ conv_in + DWConv/GELU/1x1]
  FREQ --> FREQFEAT[freq_feat: (B,freq_ch,H,W)]
  FREQFEAT --> FREQPROJ[freq_proj 1x1\nfreq_ch -> C]
  FREQFEAT --> FREQGATE[freq_gate 1x1 + Sigmoid\nfreq_ch -> C]
  FREQPROJ --> FREQMUL[freq * freq_gate]
  FREQGATE --> FREQMUL
  FREQOPT -->|No| SKIPF[skip]

  HEAD --> F0[feat0: (B,C,H,W)]
  F0 --> ADDT[Add texture prior]
  TEXMUL --> ADDT
  ADDT --> ADDF[Add freq prior (optional)]
  FREQMUL --> ADDF
  SKIPF --> ADDF

  ADDF --> RESG[RCAN Residual Groups\nn_resgroups x (n_resblocks x RCAB + conv)]
  RESG --> TRUNK[Trunk Conv3x3]
  ADDF --> GSK[Global Skip (residual)]
  TRUNK --> ADDG[Add]
  GSK --> ADDG

  ADDG --> UPS[Upsampler\nConv -> PixelShuffle(scale) -> Conv(1ch)]
  X --> BIC[Bicubic upsample x scale]
  UPS --> OUTADD[Add]
  BIC --> OUTADD
  OUTADD --> Y[SR Output\n(B,1,scale*H,scale*W)]
