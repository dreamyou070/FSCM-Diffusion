import os
import shutil
import glob

def main():
    source_folder = '../../../Comparison/InstructPix2Pix/flood_origin'
    files = os.listdir(source_folder)

    find_folder = './4_1_flood/2_nerf_road_level0123_more_youtube_fscm_styleloss_only_teacher_distill/validation_checkpoint-1800_with_style_prompt'

    save_folder = '../../../Comparison/InstructPix2Pix/ours_lora/flood'
    level_folders = {}
    for level in range(5):
        level_path = os.path.join(save_folder, f'level_{level}')
        os.makedirs(level_path, exist_ok=True)
        level_folders[level] = level_path

    for file in files:
        file_name, ext = os.path.splitext(file)
        ext = ext.lstrip('.')  # remove dot for glob pattern

        for level in range(5):

            pattern = os.path.join(find_folder, f"{file_name}_label_{level}_*.{ext}")
            matched_files = glob.glob(pattern)

            if matched_files:
                # 파일이 여럿일 경우 첫 번째만 복사
                level_file_path = matched_files[0]
                new_file_path = os.path.join(level_folders[level], file)
                shutil.copy(level_file_path, new_file_path)
                print(f"[Copied] Level {level}: {level_file_path} -> {new_file_path}")
            else:
                print(f"[Missing] No match for {file_name}_level_{level}_*.{ext}")

if __name__ == '__main__':
    main()
