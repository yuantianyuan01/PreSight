onenorth_val = [
    "scene-0194", "scene-0952", "scene-0016", "scene-0273", "scene-0272", 
    "scene-0274", "scene-0278", "scene-0980", "scene-0061", "scene-0372", 
    "scene-0381", "scene-0979", "scene-0007", "scene-0377", "scene-0983", 
    "scene-0029", "scene-0126", "scene-0977", "scene-0025", "scene-0160", 
    "scene-0054", "scene-0356", "scene-0015", "scene-0135", "scene-0344", 
    "scene-0270", "scene-0018", "scene-0362", "scene-0358", "scene-0345", 
    "scene-0968", "scene-0034", "scene-0132", "scene-0130", "scene-0191", 
    "scene-0969", "scene-0134", "scene-0957", "scene-0364", "scene-0383", 
    "scene-0190", "scene-0346", "scene-0347", "scene-0124", "scene-0368", 
    "scene-0945", "scene-0045", "scene-0961", "scene-0046", "scene-0982", 
    "scene-0151", "scene-0357", "scene-0271", "scene-0365", "scene-0060", 
    "scene-0008", "scene-0277", "scene-0359", "scene-0374", "scene-0966", 
    "scene-0960", "scene-0956", "scene-0193", "scene-0385", "scene-0154", 
    "scene-0376", "scene-0976", "scene-0043", "scene-0958", "scene-0978", 
    "scene-0371", "scene-0981", "scene-0192", "scene-0380", "scene-0984", 
    "scene-0051", "scene-0967", "scene-0059", "scene-0131", "scene-0975", 
    "scene-0014", "scene-0158", "scene-0023", "scene-0022", "scene-0355", 
    "scene-0971", "scene-0366", "scene-0949", "scene-0120", "scene-0138", 
    "scene-0367", "scene-0375", "scene-0011", "scene-0990", "scene-0003", 
    "scene-0379", "scene-0012", "scene-0159", "scene-0989", "scene-0001", 
    "scene-0963", "scene-0959", "scene-0005", "scene-0157", "scene-0275", 
    "scene-0382", "scene-0221", "scene-0028", "scene-0276", "scene-0991", 
    "scene-0150", "scene-0021", "scene-0155", "scene-0152", "scene-0006", 
    "scene-0361", "scene-0373", "scene-0363", "scene-0127", "scene-0953"
]

holland_val = [
    "scene-1045", "scene-1046", "scene-1047", "scene-1048", "scene-1049", 
    "scene-1050", "scene-1051", "scene-1052", "scene-1053", "scene-1054", 
    "scene-1055", "scene-1056", "scene-1057", "scene-1058", "scene-1074", 
    "scene-1075", "scene-1076", "scene-1077", "scene-1104", "scene-1059", 
    "scene-1060", "scene-1061", "scene-1062", "scene-1063", "scene-1064", 
    "scene-1065", "scene-1066", "scene-1067", "scene-1068", "scene-1069", 
    "scene-1070", "scene-1071", "scene-1072", "scene-1073"
]

VAL = onenorth_val + holland_val

onenorth_val_prior = [
    "scene-0055", "scene-0042", "scene-0370", "scene-0039", "scene-0047", 
    "scene-0035", "scene-0036", "scene-0044", "scene-0121", "scene-0053", 
    "scene-0057", "scene-0033", "scene-0058", "scene-0049", "scene-0378", 
    "scene-0009", "scene-0123", "scene-0056", "scene-0010", "scene-0352", 
    "scene-0048", "scene-0354", "scene-0269", "scene-0196", "scene-0050", 
    "scene-0125", "scene-0038", "scene-0350", "scene-0031", "scene-0004", 
    "scene-0139", "scene-0013", "scene-0955", "scene-0149", "scene-0052", 
    "scene-0348", "scene-0128", "scene-0962", "scene-0353", "scene-0195", 
    "scene-0017", "scene-0041", "scene-0972", "scene-0032", "scene-0351", 
    "scene-0027", "scene-0129", "scene-0030", "scene-0002", "scene-0026", 
    "scene-0268", "scene-0988", "scene-0024", "scene-0384", "scene-0133", 
    "scene-0122", "scene-0386", "scene-0019", "scene-0360", "scene-0369", 
    "scene-0349", "scene-0947", "scene-0020"
]

holland_val_prior = [
    "scene-0399", "scene-0400", "scene-0401", "scene-0402", "scene-0403", 
    "scene-0405", "scene-0406", "scene-0407", "scene-0408", "scene-0410", 
    "scene-0411", "scene-0412", "scene-0413", "scene-0414", "scene-0415", 
    "scene-0416", "scene-0417", "scene-0418", "scene-0419"
]

PRIOR = onenorth_val_prior + holland_val_prior

# Incorrect pose and map annotations according to https://github.com/nutonomy/nuscenes-devkit
POSE_FAIL_SCENES = [
    'scene-0499', 'scene-0515', 'scene-0517'
]
MAP_FAIL_SCENES = ['scene-501', 'scene-0502']