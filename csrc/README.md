Code comes from https://github.com/Dao-AILab/flash-attention/tree/main repo, the commit id:85881f5 annotation: Bump to v2.5.7  
   cp csrc/flash_attn/src/* .  
   rm flash_bwd_*  
   api float type -> double type
   api int type -> int64_t type  
   api return vector<Tensor> -> Tensor  
   api add class namespace: FTFlashAttn::  
   api c10::optional<at::Tensor> -> const c10::optional<at::Tensor>  
   api c10::optional<const at::Tensor> -> const c10::optional<at::Tensor>  
   api set_params_alibi argument:c10::optional<at::Tensor> &alibi_slopes_ -> const c10::optional<at::Tensor> &alibi_slopes_  
