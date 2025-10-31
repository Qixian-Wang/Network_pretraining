    # def _fano_factor(self):
    #     fano_factor_list = []
    #     spike_train = self.firing_rate_result['spike_trains']
    #     for spike_train in spike_train:
    #         fano_factor = est.fanofactor(spike_train)
    #         fano_factor_list.append(fano_factor)
    #     return fano_factor_list

    # def _isi_cv2_lv(self):
    #     isi_seconds_list = []
    #     cv2_vals_list = []
    #     lv_vals_list = []
    #     spike_train = self.firing_rate_result['spike_trains']
    #     for spike_train_per_sample in spike_train:
    #         isi_per_sample = []
    #         cv2_per_sample = []
    #         lv_per_sample = []
    #         for channel in spike_train_per_sample:
    #             if len(channel) < 3:
    #                 continue
    #             isi = est.isi(channel)
    #             cv2 = est.cv2(isi)
    #             lv = est.lv(isi)

    #             isi_per_sample.append(isi)
    #             cv2_per_sample.append(cv2)
    #             lv_per_sample.append(lv)

    #         isi_seconds_list.append(isi_per_sample)
    #         cv2_vals_list.append(cv2_per_sample)
    #         lv_vals_list.append(lv_per_sample)
    #     return isi_seconds_list, cv2_vals_list, lv_vals_list

    