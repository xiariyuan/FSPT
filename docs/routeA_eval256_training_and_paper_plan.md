# Route A锛圕oTracker3 + FSPT Refiner锛? eval256 璁粌涓庡彂鏂囨墽琛岃鍒?
杩欎唤璁″垝鐨勭洰鏍囨槸锛?*鎶婅缁?楠岃瘉鍙ｅ緞涓?CoTracker 瀹樻柟 TAP-Vid DAVIS strided 璇勬祴瀵归綈锛坋val256锛?*锛屽苟鍦ㄦ鍩虹涓婂仛涓€濂楄兘鍐欒繘璁烘枃鐨勩€佸彲澶嶇幇鐨勫疄楠岀煩闃碉紙baseline + 鏂规硶 + ablations + long-occlusion 涓撻」锛夈€?
---

## 0. 鍏堟妸鈥滃彛寰勨€濋攣姝伙紙蹇呴』鍋氾級

### 0.1 涓轰粈涔堣鐢?eval256
CoTracker 瀹樻柟 `TapVidDataset` 榛樿浼氭妸 DAVIS 瑙嗛 resize 鍒?`256脳256` 鍚庡啀璺?tracker 骞惰瘎娴嬶紙瑙?`baselines/cotracker/cotracker/datasets/tap_vid_datasets.py` 鐨?`resize_to=[256, 256]` 榛樿鍊硷級銆?
鍥犳锛?- 鐢?`480脳854` 鍘熷鍒嗚鲸鐜囪瘎娴嬩細寰楀埌鏄庢樉鏇翠綆鐨?`AJ/<1px/<2px`锛?*涓嶈兘鐩存帴瀵规爣 CoTracker 瀹樻柟杈撳嚭**銆?- 鍙戞枃鏃堕渶瑕佹槑纭啓娓呮 **璇勬祴鍒嗚鲸鐜囧彛寰?*锛涜繖閲岄粯璁ら噰鐢?CoTracker 瀹樻柟 eval256 浣滀负涓诲彛寰勩€?
### 0.2 瀹樻柟 CoTracker baseline锛堝浐瀹氶敋鐐癸級
纭瀹樻柟 eval 宸叉湁缁撴灉锛堜緥锛夛細
- `outputs/cotr_eval_offline_20260302_231857/result_eval_.json`
- `average_jaccard 鈮?0.7043`

濡傛灉闇€瑕侀噸鏂拌窇锛堟湇鍔″櫒锛夛細
```bash
cd /gemini/code/FSPT
/root/miniconda3/bin/python baselines/cotracker/cotracker/evaluation/evaluate.py \
  dataset_name=tapvid_davis_strided \
  dataset_root=/gemini/code/datasets \
  checkpoint=/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth \
  exp_dir=/gemini/code/FSPT/outputs/cotr_eval_offline_<time> \
  offline_model=True window_len=60
```

### 0.3 鎴戜滑璁粌/楠岃瘉涔熷繀椤?eval256
鏍稿績閰嶇疆鍘熷垯锛?- `data.val.resolution: [256, 256]`
- `evaluation.metric_resolution_mode: input`
- 浠嶇劧淇濈暀 `aux_metric_resolution_mode: original` 鍋?debug锛堜笉浣滀负 paper 涓绘寚鏍囷級

鏈?repo 宸茬粰鍑轰袱涓?eval256 閰嶇疆锛堝彲鐩存帴鐢級锛?- baseline锛堟棤棰戝煙锛夛細`configs/fspt_routeA_ms_corr_fnet_c2_corrproj_query_aggressive_eval256.yaml`
- 棰戝煙鏂规硶锛堜綘鐨?idea锛夛細`configs/fspt_routeA_freqguided_ms_corr_v2_velinv_eval256.yaml`

---

## 1. 璁粌璁″垝锛堝伐绋嬪彲鎵ц锛?
涓嬮潰鎸夆€滃厛鍙瘮 鈫?鍐嶆湁鏁?鈫?鍐嶅彲鍐欒鏂団€濈殑椤哄簭銆?
### Phase A锛欱aseline run锛堟棤棰戝煙锛?鐩殑锛氭嬁鍒颁竴鏉″畬鍏ㄥ鏍?CoTracker 鐨勨€滄垜浠嚜宸辩殑 refiner baseline鈥濇洸绾匡紝浣滀负鍚庣画 ablation 鐨勫鐓х粍銆?
鍚姩锛堟湇鍔″櫒锛夛細
```bash
cd /gemini/code/FSPT
tmux new -s routeA_base_eval256 -d "/root/miniconda3/bin/python train.py \
  --config configs/fspt_routeA_ms_corr_fnet_c2_corrproj_query_aggressive_eval256.yaml \
  --experiment.name routeA_base_eval256_$(date +%Y%m%d_%H%M%S) 2>&1 | tee outputs/routeA_base_eval256.log"
```

鍋滄鏉′欢锛堝缓璁級锛?- `AJ_delta` 杩炵画 6 娆?eval 娌℃湁鎻愬崌 >= `2e-4`锛坧lateau guard锛?- 鎴栬€?`AJ_delta` 鏄庢樉杞礋骞舵寔缁紙閬垮厤娴垂绠楀姏锛?
### Phase B锛氭柟娉?run锛堥鍩熷紩瀵艰瀺鍚堬級
鐩殑锛氬彧鏀瑰彉鈥滈鍩熷紩瀵艰瀺鍚堚€濊繖涓€鐐癸紝鍏朵綑淇濇寔鐩稿悓锛屽舰鎴?paper-friendly 鐨勫姣斻€?
鍚姩锛堟湇鍔″櫒锛夛細
```bash
cd /gemini/code/FSPT
tmux new -s routeA_freq_eval256 -d "/root/miniconda3/bin/python train.py \
  --config configs/fspt_routeA_freqguided_ms_corr_v2_velinv_eval256.yaml \
  --experiment.name routeA_freq_eval256_$(date +%Y%m%d_%H%M%S) 2>&1 | tee outputs/routeA_freq_eval256.log"
```

### Phase C锛歀ong-occlusion 涓撻」锛堝啓璁烘枃鐨勨€滀富鏁呬簨鈥濓級
鐜板疄鎯呭喌锛氬湪寮?baseline锛圕oTracker3锛変笂杩芥眰鏁翠綋 `AJ_delta` 寰堥毦鍋氬埌澶у箙鎻愬崌锛涗絾鍦?**闀块伄鎸￠噸瀹氫綅** 鍦烘櫙鏈夋洿鍙鐨勭┖闂淬€?
绛栫暐锛堟帹鑽愰『搴忥級锛?1) 淇濇寔 `AJ_delta >= 0`锛坉o-no-harm锛?2) 鎶婁紭鍖栫洰鏍囪浆鍚?`AJ_longocc30_delta` / `AJ_longocc20_delta`
3) 鐢ㄦ俯鍜岀殑 long-occlusion 鏁版嵁澧炲己鎴栬Е鍙戠瓥鐣ユ彁楂樺彲淇鎬?
锛堣繖閮ㄥ垎濡傛灉浣犵‘璁よ浠?long-occlusion 涓轰富绾匡紝鎴戝缓璁崟鐙紑涓€濂?`configs/fspt_routeA_longocc_*_eval256.yaml`锛屽苟璁?early stopping 鐩戞帶 longocc30_delta銆傦級

---

## 2. 姣忔姹囨姤鏍煎紡锛堢粺涓€鐪嬫澘锛?
鐢ㄨ剼鏈 `outputs/<exp>/epoch_metrics.jsonl`锛?```bash
cd /gemini/code/FSPT
/root/miniconda3/bin/python scripts/report_routeA_metrics.py --exp-dir outputs/<exp_name>
```

寤鸿姣忔姹囨姤鍥哄畾鍖呭惈锛?- `AJ_base / AJ / AJ_delta`
- `OA_base / OA / OA_delta`
- `avg_error_px_base / avg_error_px / delta`
- `AJ_longocc20_delta / AJ_longocc30_delta`锛堝鏋滃紑鍚級
- 鈥滃綋鍓嶆渶浣?epoch鈥濓紙鎸?`AJ_delta` 鍜?`AJ_longocc30_delta` 鍚勮嚜璁板綍锛?
---

## 3. 鍙戞枃璁″垝锛堝彲浜や粯鐗╂竻鍗曪級

### 3.1 浣犺繖鏉＄嚎鏈€瀹规槗鍐欐垚鐨勮鏂囩偣
鎺ㄨ崘涓荤嚎锛?*Frequency-guided multi-scale correspondence refinement for long-occlusion relocalization**锛堝洿缁曗€滈暱閬尅閲嶅畾浣嶁€濈殑 improvement + 鍒嗘瀽锛夈€?
鏍稿績鍗栫偣寤鸿鍐欐垚 3 鏉★細
1) multi-scale local correlation锛坈oarse+fine锛?2) frequency-guided fusion锛堜綘鐨勫垱鏂扮偣锛宲aper knob锛?3) do-no-harm guardrail锛堜繚璇佷笉浼氱牬鍧忓己 baseline锛?
### 3.2 璁烘枃蹇呴』鍑嗗鐨勫疄楠?- 涓昏〃锛欴AVIS strided eval256锛宍AJ/OA/<1/<2/<4`锛宐ase vs refined vs delta
- 瀛愰泦琛細longocc20/30锛堣嚦灏戜竴涓槇鍊硷級
- Ablation锛氬幓鎺?frequency_guided_fusion锛堝彧淇濈暀 confidence fusion锛?- 娑堣瀺锛氬彧 coarse / 鍙?fine / coarse+fine
- 閫熷害/鏄惧瓨锛氳缁?鎺ㄧ悊寮€閿€锛堣嚦灏戠粰鐩稿鍊嶆暟锛?- 鍙鍖栵細鎸?3鈥? 涓?long-occlusion failure case锛坆ase 澶辫触浣?refined 鎴愬姛锛?
### 3.3 鈥滃涓嶅鍐欒鏂団€濈殑纭棬妲涳紙寤鸿锛?- 鐩爣 1锛堢ǔ锛夛細`AJ_delta >= +0.0005` 涓旂ǔ瀹氾紙>3 娆?eval 涓嶅洖閫€锛?- 鐩爣 2锛堟晠浜嬶級锛歚AJ_longocc30_delta >= +0.005`锛堣繖涓噺绾ф墠鏇村儚鈥滆础鐚€濓級
- 濡傛灉鎬讳綋 delta 寰堝皬浣?longocc 鎻愬崌鏄庢樉锛氫緷鐒惰兘鍐欙紝浣嗘晠浜嬭鏄庣‘鈥滀笓鏀婚暱閬尅閲嶅畾浣嶁€?

