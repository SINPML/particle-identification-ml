import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hipopy.hipopy as hp
import glob
from collections import defaultdict

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


# Constants for CLAS12 detector types

DC_DETECTOR = 6
FTOF_DETECTOR = 12
ECAL_DETECTOR = 7
PCAL_DETECTOR = 7  # PCAL is part of ECAL with specific layers
ECIN_DETECTOR = 7
ECOUT_DETECTOR = 7
HTCC_DETECTOR = 15
LTCC_DETECTOR = 16

# Layer definitions for calorimeter
PCAL_LAYER = 1
ECIN_LAYER = 4
ECOUT_LAYER = 7


def create_event_index_map(pindex_array, *other_arrays):
   
    index_map = defaultdict(list)
    for idx, pidx in enumerate(pindex_array):
        if pidx >= 0:
            index_map[pidx].append(idx)
    return index_map

def find_best_calo_event(pindex_map, detector_array, energy_array, layer_array=None, 
                         extra_arrays=None, detector_preference=[ECAL_DETECTOR]):
    
    best_hits = {}
    
    for pidx, hit_indices in pindex_map.items():
        best_energy = -1
        best_idx = None
        
        for hit_idx in hit_indices:
            detector = detector_array[hit_idx]
            energy = energy_array[hit_idx]
            
            # Prefer certain detectors and higher energy
            if detector in detector_preference and energy > best_energy:
                best_energy = energy
                best_idx = hit_idx
        
        if best_idx is not None:
            hit_info = {
                'calo_idx': best_idx,
                'detector': detector_array[best_idx],
                'energy': energy_array[best_idx],
            }
            
            # Add layer info if available
            if layer_array is not None:
                hit_info['layer'] = layer_array[best_idx]
            
            # Add extra arrays if provided
            if extra_arrays:
                for key, arr in extra_arrays.items():
                    if best_idx < len(arr):
                        hit_info[key] = arr[best_idx]
            
            best_hits[pidx] = hit_info
    
    return best_hits

def find_scint_event(pindex_map, detector_array, energy_array, time_array, path_array,
                     extra_arrays=None):
    
    best_hits = {}
    
    for pidx, hit_indices in pindex_map.items():
        best_energy = -1
        best_idx = None
        
        for hit_idx in hit_indices:
            energy = energy_array[hit_idx]
            if energy > best_energy:
                best_energy = energy
                best_idx = hit_idx
        
        if best_idx is not None:
            hit_info = {
                'detector': detector_array[best_idx],
                'energy': energy_array[best_idx],
                'time': time_array[best_idx],
                'path': path_array[best_idx],
            }
            
            if extra_arrays:
                for key, arr in extra_arrays.items():
                    if best_idx < len(arr):
                        hit_info[key] = arr[best_idx]
            
            best_hits[pidx] = hit_info
    
    return best_hits

def find_cherenkov_event(pindex_map, detector_array, nphe_array, time_array, path_array):
  
    htcc_hits = {}
    ltcc_hits = {}
    
    for pidx, hit_indices in pindex_map.items():
        for hit_idx in hit_indices:
            detector = detector_array[hit_idx]
            hit_info = {
                'nphe': nphe_array[hit_idx],
                'time': time_array[hit_idx],
                'path': path_array[hit_idx],
            }
            
            if detector == HTCC_DETECTOR:
                if pidx not in htcc_hits or nphe_array[hit_idx] > htcc_hits[pidx]['nphe']:
                    htcc_hits[pidx] = hit_info
            elif detector == LTCC_DETECTOR:
                if pidx not in ltcc_hits or nphe_array[hit_idx] > ltcc_hits[pidx]['nphe']:
                    ltcc_hits[pidx] = hit_info
    
    return htcc_hits, ltcc_hits

def find_track_event(pindex_map, detector_array, sector_array, status_array, 
                     q_array, chi2_array, ndf_array):
   
    tracks = {}
    
    for pidx, hit_indices in pindex_map.items():
        if hit_indices:
            # Take first track
            idx = hit_indices[0]
            tracks[pidx] = {
                'detector': detector_array[idx],
                'sector': sector_array[idx],
                'status': status_array[idx],
                'q': q_array[idx],
                'chi2': chi2_array[idx],
                'NDF': ndf_array[idx],
            }
    
    return tracks

def find_traj_event(pindex_map, detector_array, layer_array, edge_array):
  
    traj_info = defaultdict(dict)
    
    for pidx, hit_indices in pindex_map.items():
        for idx in hit_indices:
            detector = detector_array[idx]
            layer = layer_array[idx]
            edge = edge_array[idx]
            
            if detector == DC_DETECTOR:
                traj_info[pidx][f'traj_DC_layer{layer}_edge'] = edge
            elif detector == ECAL_DETECTOR:
                traj_info[pidx][f'traj_ECAL_edge'] = edge
            elif detector == FTOF_DETECTOR:
                traj_info[pidx][f'traj_FTOF_edge'] = edge
    
    return traj_info



def extract_all_features_from_hipo(files, step=2000):
    
    features_list = []
    
    banks_to_read = [
        "REC::Particle",
        "REC::Calorimeter",
        "REC::CaloExtras",
        "REC::Scintillator", 
        "REC::ScintExtras",
        "REC::Cherenkov",
        "REC::Track",
        "REC::Traj",
        "MC::GenMatch",
        "MC::Particle"
    ]
    
    print(f"Reading banks: {banks_to_read}")
    
    batch_num = 0
    
    for batch in hp.iterate(files, banks=banks_to_read, step=step):
        
        batch_num += 1
        print(f"Processing batch {batch_num}...")
        
        
        rec_pid_evts = batch.get("REC::Particle_pid", [])
        rec_px_evts = batch.get("REC::Particle_px", [])
        rec_py_evts = batch.get("REC::Particle_py", [])
        rec_pz_evts = batch.get("REC::Particle_pz", [])
        
     
        rec_vx_evts = batch.get("REC::Particle_vx", None)
        rec_vy_evts = batch.get("REC::Particle_vy", None)
        rec_vz_evts = batch.get("REC::Particle_vz", None)
        rec_charge_evts = batch.get("REC::Particle_charge", None)
        rec_beta_evts = batch.get("REC::Particle_beta", None)
        rec_chi2pid_evts = batch.get("REC::Particle_chi2pid", None)
        rec_status_evts = batch.get("REC::Particle_status", None)
        

        calo_pindex_evts = batch.get("REC::Calorimeter_pindex", None)
        calo_detector_evts = batch.get("REC::Calorimeter_detector", None)
        calo_layer_evts = batch.get("REC::Calorimeter_layer", None)
        calo_energy_evts = batch.get("REC::Calorimeter_energy", None)
        calo_time_evts = batch.get("REC::Calorimeter_time", None)
        calo_path_evts = batch.get("REC::Calorimeter_path", None)
        calo_chi2_evts = batch.get("REC::Calorimeter_chi2", None)
        calo_x_evts = batch.get("REC::Calorimeter_x", None)
        calo_y_evts = batch.get("REC::Calorimeter_y", None)
        calo_z_evts = batch.get("REC::Calorimeter_z", None)
        calo_lu_evts = batch.get("REC::Calorimeter_lu", None)
        calo_lv_evts = batch.get("REC::Calorimeter_lv", None)
        calo_lw_evts = batch.get("REC::Calorimeter_lw", None)
        calo_m2u_evts = batch.get("REC::Calorimeter_m2u", None)
        calo_m2v_evts = batch.get("REC::Calorimeter_m2v", None)
        calo_m2w_evts = batch.get("REC::Calorimeter_m2w", None)
        
        calo_extra_size_evts = batch.get("REC::CaloExtras_size", None)
        calo_extra_eu_evts = batch.get("REC::CaloExtras_recEU", None)
        calo_extra_ev_evts = batch.get("REC::CaloExtras_recEV", None)
        calo_extra_ew_evts = batch.get("REC::CaloExtras_recEW", None)
        
        scint_pindex_evts = batch.get("REC::Scintillator_pindex", None)
        scint_detector_evts = batch.get("REC::Scintillator_detector", None)
        scint_energy_evts = batch.get("REC::Scintillator_energy", None)
        scint_time_evts = batch.get("REC::Scintillator_time", None)
        scint_path_evts = batch.get("REC::Scintillator_path", None)
        scint_chi2_evts = batch.get("REC::Scintillator_chi2", None)
        
        scint_extra_dedx_evts = batch.get("REC::ScintExtras_dedx", None)
        scint_extra_size_evts = batch.get("REC::ScintExtras_size", None)
        
        cher_pindex_evts = batch.get("REC::Cherenkov_pindex", None)
        cher_detector_evts = batch.get("REC::Cherenkov_detector", None)
        cher_nphe_evts = batch.get("REC::Cherenkov_nphe", None)
        cher_time_evts = batch.get("REC::Cherenkov_time", None)
        cher_path_evts = batch.get("REC::Cherenkov_path", None)
        
        track_pindex_evts = batch.get("REC::Track_pindex", None)
        track_detector_evts = batch.get("REC::Track_detector", None)
        track_sector_evts = batch.get("REC::Track_sector", None)
        track_status_evts = batch.get("REC::Track_status", None)
        track_q_evts = batch.get("REC::Track_q", None)
        track_chi2_evts = batch.get("REC::Track_chi2", None)
        track_ndf_evts = batch.get("REC::Track_NDF", None)
        
        traj_pindex_evts = batch.get("REC::Traj_pindex", None)
        traj_detector_evts = batch.get("REC::Traj_detector", None)
        traj_layer_evts = batch.get("REC::Traj_layer", None)
        traj_edge_evts = batch.get("REC::Traj_edge", None)
        
       
        genmatch_pindex_evts = batch.get("MC::GenMatch_pindex", None)
        genmatch_mcindex_evts = batch.get("MC::GenMatch_mcindex", None)
        
        mc_pid_evts = batch.get("MC::Particle_pid", None)
	mc_px_evts = batch.get("MC::Particle_px", None)
	mc_py_evts = batch.get("MC::Particle_py", None)
	mc_pz_evts = batch.get("MC::Particle_pz", None)
        
       
        n_events = len(rec_pid_evts)
        
        for ev in range(n_events):
            
          
            pid_arr = rec_pid_evts[ev]
            px_arr = rec_px_evts[ev]
            py_arr = rec_py_evts[ev]
            pz_arr = rec_pz_evts[ev]
            
            n_particles = len(pid_arr)
            
           
            if rec_vx_evts is not None:
                vx_arr = rec_vx_evts[ev]
                vy_arr = rec_vy_evts[ev]
                vz_arr = rec_vz_evts[ev]
            else:
                vx_arr = np.zeros(n_particles)
                vy_arr = np.zeros(n_particles)
                vz_arr = np.zeros(n_particles)
            
           
            if rec_charge_evts is not None:
                charge_arr = rec_charge_evts[ev]
            else:
                charge_arr = np.zeros(n_particles)
                
            if rec_beta_evts is not None:
                beta_arr = rec_beta_evts[ev]
            else:
                beta_arr = np.zeros(n_particles)
                
            if rec_chi2pid_evts is not None:
                chi2pid_arr = rec_chi2pid_evts[ev]
            else:
                chi2pid_arr = np.zeros(n_particles)
                
            if rec_status_evts is not None:
                status_arr = rec_status_evts[ev]
            else:
                status_arr = np.zeros(n_particles)
            
          
            calo_best = {}
            if calo_pindex_evts is not None:
                calo_pidx = calo_pindex_evts[ev]
                calo_det = calo_detector_evts[ev]
                calo_eng = calo_energy_evts[ev]
                calo_lay = calo_layer_evts[ev] if calo_layer_evts is not None else None
                
                extra_arrays = {}
                if calo_time_evts is not None:
                    extra_arrays['time'] = calo_time_evts[ev]
                if calo_path_evts is not None:
                    extra_arrays['path'] = calo_path_evts[ev]
                if calo_chi2_evts is not None:
                    extra_arrays['chi2'] = calo_chi2_evts[ev]
                if calo_x_evts is not None:
                    extra_arrays['x'] = calo_x_evts[ev]
                if calo_y_evts is not None:
                    extra_arrays['y'] = calo_y_evts[ev]
                if calo_z_evts is not None:
                    extra_arrays['z'] = calo_z_evts[ev]
                if calo_lu_evts is not None:
                    extra_arrays['lu'] = calo_lu_evts[ev]
                if calo_lv_evts is not None:
                    extra_arrays['lv'] = calo_lv_evts[ev]
                if calo_lw_evts is not None:
                    extra_arrays['lw'] = calo_lw_evts[ev]
                if calo_m2u_evts is not None:
                    extra_arrays['m2u'] = calo_m2u_evts[ev]
                if calo_m2v_evts is not None:
                    extra_arrays['m2v'] = calo_m2v_evts[ev]
                if calo_m2w_evts is not None:
                    extra_arrays['m2w'] = calo_m2w_evts[ev]
                    
                
                calo_map = create_event_index_map(calo_pidx)
                calo_best = find_best_calo_event(calo_map, calo_det, calo_eng, calo_lay, extra_arrays)
            
            # Scintillator mapping
            scint_best = {}
            if scint_pindex_evts is not None:
                scint_pidx = scint_pindex_evts[ev]
                scint_det = scint_detector_evts[ev]
                scint_eng = scint_energy_evts[ev]
                scint_t = scint_time_evts[ev]
                scint_path = scint_path_evts[ev]
                
                extra_arrays = {}
                if scint_chi2_evts is not None:
                    extra_arrays['chi2'] = scint_chi2_evts[ev]
                if scint_extra_dedx_evts is not None and ev < len(scint_extra_dedx_evts):
                    extra_arrays['dedx'] = scint_extra_dedx_evts[ev]
                if scint_extra_size_evts is not None and ev < len(scint_extra_size_evts):
                    extra_arrays['size'] = scint_extra_size_evts[ev]
                
                scint_map = create_event_index_map(scint_pidx)
                scint_best = find_scint_event(scint_map, scint_det, scint_eng, scint_t, scint_path, extra_arrays)
            
            # Cherenkov mapping
            htcc_best = {}
            ltcc_best = {}
            if cher_pindex_evts is not None:
                cher_pidx = cher_pindex_evts[ev]
                cher_det = cher_detector_evts[ev]
                cher_nphe = cher_nphe_evts[ev]
                cher_t = cher_time_evts[ev] if cher_time_evts is not None else np.zeros(len(cher_pidx))
                cher_path = cher_path_evts[ev] if cher_path_evts is not None else np.zeros(len(cher_pidx))
                
                cher_map = create_event_index_map(cher_pidx)
                htcc_best, ltcc_best = find_cherenkov_event(cher_map, cher_det, cher_nphe, cher_t, cher_path)
            
            # Track mapping
            track_best = {}
            if track_pindex_evts is not None:
                track_pidx = track_pindex_evts[ev]
                track_det = track_detector_evts[ev]
                track_sec = track_sector_evts[ev]
                track_stat = track_status_evts[ev]
                track_q = track_q_evts[ev]
                track_chi2 = track_chi2_evts[ev]
                track_ndf = track_ndf_evts[ev]
                
                track_map = create_event_index_map(track_pidx)
                track_best = find_track_event(track_map, track_det, track_sec, track_stat, 
                                              track_q, track_chi2, track_ndf)
            
            # Trajectory mapping
            traj_best = {}
            if traj_pindex_evts is not None:
                traj_pidx = traj_pindex_evts[ev]
                traj_det = traj_detector_evts[ev]
                traj_lay = traj_layer_evts[ev]
                traj_edge = traj_edge_evts[ev]
                
                traj_map = create_event_index_map(traj_pidx)
                traj_best = find_traj_event(traj_map, traj_det, traj_lay, traj_edge)
            
            # MC matching for this event
            mc_match = {}
            mc_match = {}
	    if genmatch_pindex_evts is not None:
            gm_pidx = genmatch_pindex_evts[ev]
            gm_mcidx = genmatch_mcindex_evts[ev]
    
            for i, pidx in enumerate(gm_pidx):
            	if pidx >= 0:
            		mc_idx = gm_mcidx[i]
           		mc_match[pidx] = {
                	'index': mc_idx,
                	'px': mc_px_evts[ev][mc_idx] if mc_px_evts is not None else 0,
                	'py': mc_py_evts[ev][mc_idx] if mc_py_evts is not None else 0,
                	'pz': mc_pz_evts[ev][mc_idx] if mc_pz_evts is not None else 0
            }
            
   
            for pidx in range(n_particles):
                
             
                px = px_arr[pidx]
                py = py_arr[pidx]
                pz = pz_arr[pidx]
                
                p = np.sqrt(px**2 + py**2 + pz**2)
                pt = np.sqrt(px**2 + py**2)
                
                if p > 1e-12:
                    cos_theta = np.clip(pz / p, -1.0, 1.0)
                    theta = np.arccos(cos_theta) * 180 / np.pi
                else:
                    theta = 0
                
                phi = np.arctan2(py, px) * 180 / np.pi
                
                # MC truth
                mc_pid = 0
		is_electron_mc = 0
		mc_px = mc_py = mc_pz = 0.0

		if pidx in mc_match and mc_pid_evts is not None:
    			mc_info = mc_match[pidx]
    			mc_idx = mc_info['index']
    
    			if mc_idx < len(mc_pid_evts[ev]):
       				mc_pid = mc_pid_evts[ev][mc_idx]
        			is_electron_mc = 1 if abs(mc_pid) == 11 else 0
        			mc_px = mc_info['px']
        			mc_py = mc_info['py']
        			mc_pz = mc_info['pz']
                
              
                features = {
                    # REC::Particle basics
                    'rec_pid': pid_arr[pidx],
                    'rec_charge': charge_arr[pidx],
                    'rec_beta': beta_arr[pidx],
                    'rec_chi2pid': chi2pid_arr[pidx],
                    'rec_status': status_arr[pidx],
                    'p': p,
                    'pt': pt,
                    'theta': theta,
                    'phi': phi,
                    'vx': vx_arr[pidx],
                    'vy': vy_arr[pidx],
                    'vz': vz_arr[pidx],
                    'px': px,
                    'py': py,
                    'pz': pz,
			
                    
                    # MC truth
                    'mc_pid': mc_pid,
                    'is_electron_mc': is_electron_mc,
		    'mc_px': mc_px,    
    		    'mc_py': mc_py,    
    		    'mc_pz': mc_pz,   
                }
                
                # Add calorimeter features
                if pidx in calo_best:
                    calo = calo_best[pidx]
                    features.update({
                        'calo_energy': calo.get('energy', 0),
                        'calo_time': calo.get('time', 0),
                        'calo_path': calo.get('path', 0),
                        'calo_chi2': calo.get('chi2', 0),
                        'calo_x': calo.get('x', 0),
                        'calo_y': calo.get('y', 0),
                        'calo_z': calo.get('z', 0),
                        'calo_lu': calo.get('lu', 0),
                        'calo_lv': calo.get('lv', 0),
                        'calo_lw': calo.get('lw', 0),
                        'calo_m2u': calo.get('m2u', 0),
                        'calo_m2v': calo.get('m2v', 0),
                        'calo_m2w': calo.get('m2w', 0),
                        'calo_size': calo.get('size', 0),
                        'calo_eu': calo.get('recEU', 0),
                        'calo_ev': calo.get('recEV', 0),
                        'calo_ew': calo.get('recEW', 0),
                        'calo_layer': calo.get('layer', 0),
                    })
                    
                    # Derived features
                    if p > 0:
                        features['E_over_P'] = calo.get('energy', 0) / p
                    else:
                        features['E_over_P'] = 0
                
                # Add scintillator features
                if pidx in scint_best:
                    scint = scint_best[pidx]
                    features.update({
                        'scint_energy': scint.get('energy', 0),
                        'scint_time': scint.get('time', 0),
                        'scint_path': scint.get('path', 0),
                        'scint_chi2': scint.get('chi2', 0),
                        'scint_dedx': scint.get('dedx', 0),
                        'scint_size': scint.get('size', 0),
                    })
                    
                    # Calculate beta from TOF
                    if scint.get('path', 0) > 0 and scint.get('time', 0) != 0:
                        features['scint_beta'] = scint['path'] / (abs(scint['time']) * 29.979)
                    else:
                        features['scint_beta'] = 0
                
                # Add Cherenkov features
                if pidx in htcc_best:
                    features.update({
                        'htcc_nphe': htcc_best[pidx]['nphe'],
                        'htcc_time': htcc_best[pidx]['time'],
                        'htcc_path': htcc_best[pidx]['path'],
                    })
                
                if pidx in ltcc_best:
                    features.update({
                        'ltcc_nphe': ltcc_best[pidx]['nphe'],
                        'ltcc_time': ltcc_best[pidx]['time'],
                        'ltcc_path': ltcc_best[pidx]['path'],
                    })
                
                # Add track features
                if pidx in track_best:
                    trk = track_best[pidx]
                    features.update({
                        'track_detector': trk['detector'],
                        'track_sector': trk['sector'],
                        'track_status': trk['status'],
                        'track_q': trk['q'],
                        'track_chi2': trk['chi2'],
                        'track_NDF': trk['NDF'],
                    })
                    
                    if trk['NDF'] > 0:
                        features['track_chi2_per_NDF'] = trk['chi2'] / trk['NDF']
                    else:
                        features['track_chi2_per_NDF'] = 999
                
                # Add trajectory features
                if pidx in traj_best:
                    features.update(traj_best[pidx])
                
                # Additional derived features
                if 'E_over_P' in features and 'htcc_nphe' in features:
                    features['electron_score_basic'] = features['E_over_P'] * np.log1p(features['htcc_nphe'])
                
                features_list.append(features)
        
        print(f"  Batch {batch_num} complete. Total particles extracted: {len(features_list)}")
    
    print(f"\nTotal particles extracted: {len(features_list)}")
    
    return pd.DataFrame(features_list)

----------------------------------------------------------

def main():
    
    hipo_files = glob.glob("*.hipo")
    
    if not hipo_files:
        print("No HIPO files found!")
        return
    
    print(f"Found {len(hipo_files)} HIPO files")
    
    
    df = extract_all_features_from_hipo(hipo_files, step=2000)
    
    if len(df) == 0:
        print("No particles extracted!")
        return
    
    df = df.fillna(0)
    
    print(f"\nTotal particles: {len(df)}")
    print(f"Features: {len(df.columns)}")
    
    if 'is_electron_mc' in df.columns:
        n_electrons = df['is_electron_mc'].sum()
        print(f"\nMC Truth:")
        print(f"  Electrons: {n_electrons} ({100*n_electrons/len(df):.1f}%)")
        print(f"  Background: {len(df)-n_electrons} ({100*(len(df)-n_electrons)/len(df):.1f}%)")
    
    
    output_file = "clas12_electron_id_features.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")

if __name__ == "__main__":
    main()