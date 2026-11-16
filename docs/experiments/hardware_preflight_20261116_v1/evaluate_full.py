"""Isolated full validation: original PyTorch vs ONNX->production C++ pipeline."""
import argparse, csv, json, subprocess, time
from pathlib import Path
import h5py
import numpy as np
import torch
import onnxruntime as ort
from ktcal_paper.combined_3et import _sha256, load_stnet_checkpoint, load_matcher_checkpoint, validate_validation_h5, SCALE
from ktcal_paper.offline_experiment import spatial_view, undo_view, average_views, smooth_pixels
from ktcal_paper.offline_batch import OfflineBatchTracker
from ktcal_paper.evaluation_protocol import soft_expectation_decode, summarize_pixel_errors
from hardware.rv1106.export_matcher import export_matcher
from hardware.rv1106.verify_stnet import validate_export_manifest

VIEWS=[(0,0,False),(0,0,True),(-2,0,False),(-2,0,True),(2,0,False),(2,0,True),
       (0,-2,False),(0,-2,True),(0,2,False),(0,2,True)]
def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--h5',required=True);p.add_argument('--onnx',required=True)
    p.add_argument('--stnet',default='weights/3et/stnet_best.pt');p.add_argument('--matcher',default='weights/3et/matcher_best.pt')
    p.add_argument('--output-dir',required=True);p.add_argument('--limit',type=int,default=93)
    args=p.parse_args(); out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    h5,key,shape,_=validate_validation_h5(args.h5)
    if h5.name!='test_aug.h5' or shape!=(93,100,2,60,80) or not 1<=args.limit<=93:
        raise ValueError('exact 93x100 test_aug input required')
    manifest=json.loads(Path('weights/3et/manifest.json').read_text())
    if _sha256(h5)!=manifest['validation_h5_sha256'] or _sha256(args.stnet)!=manifest['stnet']['sha256'] or _sha256(args.matcher)!=manifest['matcher']['sha256']:
        raise ValueError('released weights/data hash mismatch')
    metadata=validate_export_manifest(args.stnet,args.onnx)
    torch.set_num_threads(4)
    model,config,divisor=load_stnet_checkpoint(args.stnet,'cpu')
    matcher,_=load_matcher_checkpoint(args.matcher,'cpu')
    tracker=OfflineBatchTracker(matcher,device='cpu',search_size=(37,32),confidence_threshold=.28,min_psr=2.55,
                               q_pos=.05,q_vel=.2,r_px=32.,fusion_alpha=.75)
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(args.onnx,sess_options=options,providers=['CPUExecutionProvider'])
    binary=out/'full_bridge';src=Path(__file__).with_name('full_bridge.cc')
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-Ihardware/rv1106',
                    'hardware/rv1106/fast_path.cc','hardware/rv1106/sequence_path.cc',str(src),'-o',str(binary)],check=True)
    export_matcher(args.matcher,out/'matcher.bin')
    # record41 diagnosis separates backend drift (~8e-5px) from same-logit
    # C++ reduction drift (~1.5e-5px). Keep the latter check tight; permit
    # 2e-4px backend drift, still requiring identical full-data decisions/P3.
    maxima={k:0. for k in ['logits','global_px','raw_px','filtered_px','same_onnx_global_px','same_onnx_stnet_ma7_px']}
    ref_global=[];actual_global=[];ref_raw=[];actual_raw=[];ref_final=[];actual_final=[];actual_stnet=[];labels_all=[]
    ref_counts={};actual_counts={};diff_tiers=0;cpp_ms=0.;onnx_ms=0.;rows=[]
    csvfile=(out/'predictions.csv').open('x');writer=csv.writer(csvfile)
    writer.writerow(['record','frame','x_px','y_px','raw_x_px','raw_y_px','tier',
                     'pytorch_x_px','pytorch_y_px','stnet_x_px','stnet_y_px','pytorch_stnet_x_px','pytorch_stnet_y_px'])
    with h5py.File(h5,'r') as handle,torch.inference_mode():
        for record in range(args.limit):
            started=time.perf_counter();raw=np.asarray(handle['frames'][record:record+1],dtype=np.float32)
            frames=torch.from_numpy(raw);cache=[];onnx_cache=[];logits=[];transforms=[];record_logit=0.
            for view in VIEWS:
                transformed=spatial_view(frames,view).contiguous();transforms.append(transformed.numpy()[0])
                lx,ly=model(transformed/divisor)
                begin=time.perf_counter();actual=session.run(None,{'raw_frames':transformed.numpy()});onnx_ms+=(time.perf_counter()-begin)*1000
                for a,b in zip(actual,[lx.numpy(),ly.numpy()]):
                    np.testing.assert_allclose(a,b,atol=2e-4,rtol=2e-4)
                    record_logit=max(record_logit,float(np.max(np.abs(a-b))))
                logits.append(np.concatenate(actual,axis=-1)[0])
                decoded=soft_expectation_decode(lx/.65,ly/.65).astype(np.float32)*SCALE
                cache.append(undo_view(decoded,view))
                onnx_decoded=soft_expectation_decode(actual[0]/.65,actual[1]/.65).astype(np.float32)*SCALE
                onnx_cache.append(undo_view(onnx_decoded,view))
            global_px=average_views(np.stack(cache),list(range(10)),method='median')
            same_onnx_px=average_views(np.stack(onnx_cache),list(range(10)),method='median')
            pytorch_stnet=smooth_pixels(global_px,'ma7')[0]
            expected,stats=tracker.track(frames,global_px,return_trace=True);filtered=smooth_pixels(expected,'ma7')
            for name,value in [('raw.bin',raw),('views.bin',transforms),('logits.bin',logits)]:
                np.asarray(value,dtype='<f4').tofile(out/name)
            bridge=json.loads(subprocess.check_output([str(binary),str(out/'matcher.bin'),str(out/'raw.bin'),str(out/'views.bin'),
                                                      str(out/'logits.bin'),str(out/'cpp.bin')],text=True))
            a=np.fromfile(out/'cpp.bin',dtype='<f8').reshape(100,9)
            errors={'logits':record_logit,'global_px':float(np.max(np.abs(a[:,:2]-global_px[0]))),
                    'raw_px':float(np.max(np.abs(a[:,2:4]-expected[0]))),'filtered_px':float(np.max(np.abs(a[:,4:6]-filtered[0]))),
                    'same_onnx_global_px':float(np.max(np.abs(a[:,:2]-same_onnx_px[0]))),
                    'same_onnx_stnet_ma7_px':float(np.max(np.abs(a[:,7:9]-smooth_pixels(same_onnx_px,'ma7')[0])))}
            for k,v in errors.items():maxima[k]=max(maxima[k],v)
            np.testing.assert_allclose(a[:,:2],same_onnx_px[0],atol=4e-5,rtol=0)
            np.testing.assert_allclose(a[:,7:9],smooth_pixels(same_onnx_px,'ma7')[0],atol=4e-5,rtol=0)
            np.testing.assert_allclose(a[:,:2],global_px[0],atol=2e-4,rtol=0)
            np.testing.assert_allclose(a[:,2:4],expected[0],atol=5e-4,rtol=0)
            np.testing.assert_allclose(a[:,4:6],filtered[0],atol=5e-4,rtol=0)
            np.testing.assert_allclose(a[:,7:9],pytorch_stnet,atol=2e-4,rtol=0)
            tier_errors=int(np.count_nonzero(a[:,6]!=stats['tiers'][0]));diff_tiers+=tier_errors
            if tier_errors or stats['small_fusion_calls'] or bridge['small_fusion_calls']:
                raise ValueError('tier mismatch or forbidden low fusion')
            for k in ['tier1','tier2','tier3','medium_refinements','medium_rollbacks','fusion_calls','small_fusion_calls','stnet_requests','border_peaks']:
                if stats[k]!=bridge[k]:raise ValueError('tracker stat mismatch '+k)
                ref_counts[k]=ref_counts.get(k,0)+stats[k];actual_counts[k]=actual_counts.get(k,0)+bridge[k]
            cpp_ms+=bridge['replay_and_cpp_ms']
            ref_global.append(global_px[0]);actual_global.append(a[:,:2]);ref_raw.append(expected[0]);actual_raw.append(a[:,2:4])
            ref_final.append(filtered[0]);actual_final.append(a[:,4:6]);actual_stnet.append(a[:,7:9]);labels_all.append(handle[key][record,:,:2].astype(np.float32))
            for t in range(100):writer.writerow([record,t,*a[t,4:6],*a[t,2:4],int(a[t,6]),
                                                *filtered[0,t],*a[t,7:9],*pytorch_stnet[t]])
            csvfile.flush();row=dict(record=record,frames=100,seconds=time.perf_counter()-started,max_errors=errors,stats=bridge)
            rows.append(row);print(json.dumps(row),flush=True)
    csvfile.close(); labels=np.stack(labels_all)
    def metric(values):return summarize_pixel_errors(np.stack(values)/SCALE,labels)
    metrics={'pytorch_stnet_ma7':metric(smooth_pixels(np.stack(ref_global),'ma7')),
             'onnx_cpp_stnet_ma7':metric(actual_stnet),
             'pytorch_ktcal_ma7':metric(ref_final),'onnx_cpp_ktcal_ma7':metric(actual_final)}
    for baseline,candidate in [('pytorch_stnet_ma7','onnx_cpp_stnet_ma7'),('pytorch_ktcal_ma7','onnx_cpp_ktcal_ma7')]:
        for key in ['p3','p5','p10']:
            if metrics[baseline][key]!=metrics[candidate][key]:raise ValueError('precision threshold parity failed '+key)
        if abs(metrics[baseline]['mean_error_px']-metrics[candidate]['mean_error_px'])>=2e-5:
            raise ValueError('aggregate mean error parity failed')
    report=dict(schema=1,passed=True,complete=args.limit==93,records=args.limit,frames=args.limit*100,
                h5_sha256=_sha256(h5),stnet_sha256=_sha256(args.stnet),matcher_sha256=_sha256(args.matcher),
                onnx_sha256=metadata['onnx_sha256'],max_errors=maxima,tier_mismatch_frames=diff_tiers,
                tolerances=dict(logits_atol=2e-4,logits_rtol=2e-4,same_onnx_cpp_px=4e-5,
                                pytorch_onnx_cpp_global_px=2e-4,raw_filtered_px=5e-4,
                                p3_p5_p10='exact',aggregate_mean_px=2e-5),
                tracker_stats=actual_counts,all_record_low_global_invariance=True,all_view_raw_inputs_byte_equal=True,
                metrics=metrics,host_timing=dict(onnx_cpu_ms_per_frame=onnx_ms/(args.limit*100),
                    cpp_replay_decode_match_kf_ma7_ms_per_frame=cpp_ms/(args.limit*100),
                    scope='single-pass CPU host only; no warmups; excludes disk, Python transform, reference and process startup; not board latency'),
                rknn_conversion_verified=False,board_verified=False,record_checks=rows)
    with (out/'report.json').open('x') as target:json.dump(report,target,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k!='record_checks'},indent=2),flush=True)
if __name__=='__main__':main()
