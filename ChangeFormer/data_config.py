class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = ""
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.label_transform = "norm"
            self.root_dir = '/content/drive/MyDrive/LEVIR-CD/'
        elif data_name == 'LEVIRplus':
            self.label_transform = "norm"
            self.root_dir = '/content/drive/MyDrive/LEVIR-CD-Plus/'
        elif data_name == 'DSIFN':
            self.label_transform = "norm"
            self.root_dir = '/content/drive/MyDrive/DSIFN_256/'
        elif data_name == 'xbd_split':
            self.label_transform = "norm"
            self.root_dir = '/content/drive/MyDrive/xbd_split/'
        elif data_name == 'COMBINED_CD':
            self.label_transform = "norm"
            self.root_dir = '/content/drive/MyDrive/COMBINED_CD'
        elif data_name == 'quick_start_LEVIR':
            self.root_dir = './samples_LEVIR/'
        elif data_name == 'quick_start_DSIFN':
            self.root_dir = './samples_DSIFN/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self
