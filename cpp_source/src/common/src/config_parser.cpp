#include "../include/config_parser.hpp"

namespace nl_video_analysis {

VideoAnalysisConfig ConfigParser::parseFromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + filepath);
    }

    VideoAnalysisConfig config;
    std::string line;
    bool in_cameras_array = false;
    CameraConfig current_camera;
    bool in_camera_object = false;

    bool in_detector_object = false;
    bool in_tracker_object = false;
    bool in_classes_array = false;
    bool in_image_encoder_object = false;

    while (std::getline(file, line)) {
        line = trim(line);

        if (line.empty() || line[0] == '#' || line.substr(0, 2) == "//") {
            continue;
        }

        if (!line.empty() && line.back() == ',') {
            line = line.substr(0, line.size() - 1);
        }

        if (line.find("\"cameras\"") != std::string::npos) {
            in_cameras_array = true;
            continue;
        }

        if (line.find("\"object_detector\"") != std::string::npos) {
            in_detector_object = true;
            continue;
        }
        if (line.find("\"tracker\"") != std::string::npos) {
            in_tracker_object = true;
            continue;
        }
        if (line.find("\"image_encoder\"") != std::string::npos) {
            in_image_encoder_object = true;
            continue;
        }

        if (in_cameras_array) {
            if (line.find('{') != std::string::npos) {
                in_camera_object = true;
                current_camera = CameraConfig();
                continue;
            }

            if (line.find('}') != std::string::npos) {
                if (in_camera_object) {
                    config.cameras.push_back(current_camera);
                    in_camera_object = false;
                }
                continue;
            }

            if (line.find(']') != std::string::npos) {
                in_cameras_array = false;
                continue;
            }

            if (in_camera_object) {
                if (line.find("\"camera_id\"") != std::string::npos) {
                    size_t colon = line.find(':');
                    if (colon != std::string::npos) {
                        current_camera.camera_id = parseString(line.substr(colon + 1));
                    }
                } else if (line.find("\"source_url\"") != std::string::npos) {
                    size_t colon = line.find(':');
                    if (colon != std::string::npos) {
                        current_camera.source_url = parseString(line.substr(colon + 1));
                    }
                } else if (line.find("\"source_type\"") != std::string::npos) {
                    size_t colon = line.find(':');
                    if (colon != std::string::npos) {
                        current_camera.source_type = parseString(line.substr(colon + 1));
                    }
                } else if (line.find("\"stream_codec\"") != std::string::npos) {
                    size_t colon = line.find(':');
                    if (colon != std::string::npos) {
                        std::string stream_codec = parseString(line.substr(colon + 1));
                        current_camera.parseCodec(stream_codec);
                    }
                }
            }
            continue;
        }

        if (in_detector_object) {
            if (line.find('}') != std::string::npos) {
                in_detector_object = false;
                continue;
            }

            if (line.find("\"classes\"") != std::string::npos) {
                // Check if it's a single-line array: "classes" : [0, 1, 2]
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    std::string array_part = line.substr(colon + 1);
                    size_t start_bracket = array_part.find('[');
                    size_t end_bracket = array_part.find(']');

                    if (start_bracket != std::string::npos && end_bracket != std::string::npos) {
                        std::string array_content = array_part.substr(start_bracket + 1, end_bracket - start_bracket - 1);
                        std::stringstream ss(array_content);
                        std::string token;

                        while (std::getline(ss, token, ',')) {
                            token = trim(token);
                            if (!token.empty()) {
                                try {
                                    config.object_detector.classes.push_back(std::stoi(token));
                                } catch (...) {
                                }
                            }
                        }
                    } else {
                        in_classes_array = true;
                    }
                }
                continue;
            }

            if (in_classes_array) {
                if (line.find(']') != std::string::npos) {
                    in_classes_array = false;
                    continue;
                }

                std::string numbers = trim(line);
                std::stringstream ss(numbers);
                std::string token;

                while (std::getline(ss, token, ',')) {
                    token = trim(token);
                    if (!token.empty() && token != "[" && token != "]") {
                        try {
                            config.object_detector.classes.push_back(parseInt(token));
                        } catch (...) {
                        }
                    }
                }
                continue;
            }

            if (line.find("\"type\"") != std::string::npos) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    config.object_detector.type = parseString(line.substr(colon + 1));
                }
            } else if (line.find("\"weights_path\"") != std::string::npos) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    config.object_detector.weights_path = parseString(line.substr(colon + 1));
                }
            } else if (line.find("\"number_of_threads\"") != std::string::npos) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    config.object_detector.number_of_threads = parseInt(line.substr(colon + 1));
                }
            } else if (line.find("\"conf_threshold\"") != std::string::npos) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    config.object_detector.conf_threshold = std::stof(trim(line.substr(colon + 1)));
                }
            } else if (line.find("\"nms_threshold\"") != std::string::npos) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    config.object_detector.nms_threshold = std::stof(trim(line.substr(colon + 1)));
                }
            }  else if (line.find("\"is_fp16\"") != std::string::npos) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    config.object_detector.is_fp16 = parseBool(line.substr(colon + 1));
                }
            }
            continue;
        }

        if(in_tracker_object)
        {
            if (line.find('}') != std::string::npos) {
                in_tracker_object = false;
                continue;
            }

            if (line.find("\"max_age\"") != std::string::npos) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    config.tracker.max_age = parseInt(line.substr(colon + 1));
                } 
            } else if (line.find("\"min_hits\"") != std::string::npos) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    config.tracker.min_hits = parseInt(line.substr(colon + 1));
                }
            } else if (line.find("\"iou_threshold\"") != std::string::npos) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    config.tracker.iou_threshold = std::stof(trim(line.substr(colon + 1)));
                }
            } 
            continue;
        }
        if(in_image_encoder_object)
        {
            if (line.find("\"model_path\"") != std::string::npos) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    config.image_encoder.model_path = parseString(line.substr(colon + 1));
                }
            } else if (line.find("\"number_of_threads\"") != std::string::npos) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    config.image_encoder.num_threads = parseInt(line.substr(colon + 1));
                }
            } else if (line.find("\"is_fp16\"") != std::string::npos) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    config.image_encoder.is_fp16 = parseBool(line.substr(colon + 1));
                }
            }
        }
        size_t colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }

        std::string key = line.substr(0, colon);
        std::string value = line.substr(colon + 1);
        key = trim(removeQuotes(key));
        value = trim(value);

        if (key == "max_connections") {
            config.max_connections = parseInt(value);
        } else if (key == "clip_length") {
            config.clip_length = parseInt(value);
        } else if (key == "sampler_type") {
            config.sampler_type = parseString(value);
        } else if (key == "sampled_frames_count") {
            config.sampled_frames_count = parseInt(value);
        } else if (key == "queue_max_size") {
            config.queue_max_size = parseInt(value);
        } else if (key == "gst_buffer_size") {
            config.gst_buffer_size = parseInt(value);
        } else if (key == "gst_drop_frames") {
            config.gst_drop_frames = parseInt(value);
        } else if (key == "gst_target_fps") {
            config.gst_target_fps = parseInt(value);
        } else if (key == "gst_frame_width") {
            config.gst_frame_width = parseInt(value);
        } else if (key == "gst_frame_height") {
            config.gst_frame_height = parseInt(value);
        }
    }

    file.close();
    return config;
}

std::string ConfigParser::trim(const std::string& str) {
    const char* whitespace = " \t\n\r";
    size_t start = str.find_first_not_of(whitespace);
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(whitespace);
    return str.substr(start, end - start + 1);
}

std::string ConfigParser::removeQuotes(const std::string& str) {
    std::string result = trim(str);
    if (result.length() >= 2 && result.front() == '"' && result.back() == '"') {
        return result.substr(1, result.length() - 2);
    }
    return result;
}

bool ConfigParser::parseBool(const std::string& value) {
    std::string v = trim(value);
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    return v == "true" || v == "1";
}

int ConfigParser::parseInt(const std::string& value) {
    return std::stoi(trim(value));
}

std::string ConfigParser::parseString(const std::string& value) {
    return removeQuotes(value);
}

}
