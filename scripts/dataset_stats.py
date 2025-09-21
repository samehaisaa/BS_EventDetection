import argparse, os
from gut_sed.stats import per_recording_stats

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--files", nargs="+", required=True)
    args = p.parse_args()
    mapping = {}
    for item in args.files:
        rid, wav, lab = item.split(",")
        mapping[rid] = {"wav": wav, "txt": lab}
    rec_df, lab_df, frm_df, ov_paths = per_recording_stats(mapping, args.out_dir)
    print(rec_df.to_string(index=False))
    print(lab_df.to_string(index=False))
    print(frm_df.to_string(index=False))
    for o in ov_paths:
        print(o["recording_id"], o["overlap_csv"])

if __name__=="__main__":
    main()
