import os
from pydantic import BaseModel, ConfigDict

class Configs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Mea type
    mea_type: str = "128_rhs"
    # mea_type: str = "512_long"

    # Stimulation parameters
    excluded_channels: list[str] = ['A-005', 'A-020', 'A-031', 'B-002', 'B-011', 'B-021', 'C-024', 'D-004', 'D-016', 'D-017', 'D-018']
    
    reading_channels: list[str] = [
                        'A-000', 'A-001', 'A-002', 'A-003', 'A-004', 'A-005', 'A-006', 'A-007', 
                        'A-008', 'A-009', 'A-010', 'A-011', 'A-012', 'A-013', 'A-014', 'A-015', 
                        'A-016', 'A-017', 'A-018', 'A-019', 'A-020', 'A-021', 'A-022', 'A-023', 
                        'A-024', 'A-025', 'A-026', 'A-027', 'A-028', 'A-029', 'A-030', 'A-031',
                        'D-000', 'D-001', 'D-002', 'D-003', 'D-004', 'D-005', 'D-006', 'D-007', 
                        'D-008', 'D-009', 'D-010', 'D-011', 'D-012', 'D-013', 'D-014', 'D-015', 
                        'D-016', 'D-017', 'D-018', 'D-019', 'D-020', 'D-021', 'D-022', 'D-023', 
                        'D-024', 'D-025', 'D-026', 'D-027', 'D-028', 'D-029', 'D-030', 'D-031',
                        'B-000', 'B-001', 'B-002', 'B-003', 'B-004', 'B-005', 'B-006', 'B-007', 
                        'B-008', 'B-009', 'B-010', 'B-011', 'B-012', 'B-013', 'B-014', 'B-015', 
                        'B-016', 'B-017', 'B-018', 'B-019', 'B-020', 'B-021', 'B-022', 'B-023', 
                        'B-024', 'B-025', 'B-026', 'B-027', 'B-028', 'B-029', 'B-030', 'B-031',
                        'C-000', 'C-001', 'C-002', 'C-003', 'C-004', 'C-005', 'C-006', 'C-007', 
                        'C-008', 'C-009', 'C-010', 'C-011', 'C-012', 'C-013', 'C-014', 'C-015', 
                        'C-016', 'C-017', 'C-018', 'C-019', 'C-020', 'C-021', 'C-022', 'C-023', 
                        'C-024', 'C-025', 'C-026', 'C-027', 'C-028', 'C-029', 'C-030', 'C-031'
                        ]
 
    session_names: list[str] = ["pretrain1"]

    # Pattern parameters
    num_pretrain_patterns: int = 4
    num_train_patterns: int = 6
    
    component: dict[int, list[int]] = {
        0: [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27],
        1: [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31],
        2: [96, 97, 98, 99, 104, 105, 106, 107, 112, 113, 114, 115, 120, 121, 122, 123],
        3: [100, 101, 102, 103, 108, 109, 110, 111, 116, 117, 118, 119, 124, 125, 126, 127],
    }

    train_pattern_dict: dict[int, list[int]] = {
        0: component[0]+component[1],
        1: component[0]+component[2],
        2: component[0]+component[3],
        3: component[1]+component[2],
        4: component[1]+component[3],
        5: component[2]+component[3],
    }
    
    pretrain_pattern_dict: dict[int, list[int]] = {
        0: component[0],
        1: component[1],
        2: component[2],
        3: component[3],
    }

    pretrain_stimulation_repetition: int = 4

    # File paths
    mea_yaml_path: str = f"/Users/aia/Desktop/cppcode/Intan-RHX/tcp/data_process/mea_topology/{mea_type}.yaml"
    data_path: str = "/Users/aia/Desktop/data/11_25_data_sample4/spike_train"
    figure_save_path: str = "figures"

    
configs = Configs()