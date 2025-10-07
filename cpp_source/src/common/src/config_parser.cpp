#include "../include/config_parser.hpp"
#include <iostream>
#include <algorithm>

namespace nl_video_analysis {

MediaProcessorConfig ConfigParser::parseFromFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + filepath);
    }

    MediaProcessorConfig config;
    std::string line;
    bool in_cameras_array = false;
    CameraConfig current_camera;
    bool in_camera_object = false;

    while (std::getline(file, line)) {
        line = trim(line);

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#' || line.substr(0, 2) == "//") {
            continue;
        }

        // Remove trailing commas
        if (!line.empty() && line.back() == ',') {
            line = line.substr(0, line.size() - 1);
        }

        // Check for cameras array
        if (line.find("\"cameras\"") != std::string::npos) {
            in_cameras_array = true;
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
                }
            }
            continue;
        }

        // Parse top-level config values
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
        } else if (key == "frames_per_clip") {
            config.frames_per_clip = parseInt(value);
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
