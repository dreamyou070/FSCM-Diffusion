import os


def main():

    folder = "./FSCM_Snow"
    csv_dir = os.path.join(folder, 'data.csv')
    if os.path.exists(csv_dir):
        os.remove(csv_dir)

    file_list = os.listdir(folder)

    elem = ['input_image', 'edit_prompt', 'edited_image', 'label', 'depthmap', 'normalmap']
    total = [elem]
    for file in file_list:
        if 'input.png' in file :
            input_img_dir = file
            prompt_dir = os.path.join(folder, file.replace('_input.png', '_edit_prompt.txt'))
            with open(prompt_dir, 'r',) as f:
                edit_prompt = f.readlines()[0]
            edited_img_dir = file.replace('_input.png', '_edit.png')
            depthmap_dir = file.replace('_input.png', '_input_depth.png')
            normalmap_dir = file.replace('_input.png', '_input_normalmap.png')
            level_dir = os.path.join(folder, file.replace('_input.png', '_label.txt'))
            with open(level_dir, 'r') as f:
                level = f.read()
            total.append([input_img_dir, edit_prompt, edited_img_dir, level, depthmap_dir, normalmap_dir])

    with open(csv_dir, 'w') as f:
        for elem in total:
            f.write(f'{elem[0]},{elem[1]},{elem[2]},{elem[3]},{elem[4]},{elem[5]}\n')

    len_data = len(total)
    print(f'len_data = {len_data}')

if __name__ == '__main__':
    main()
