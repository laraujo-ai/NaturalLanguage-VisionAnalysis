
We'll have the mediaprocessing block which will : 

- Accept multiple connections and create one StreamHandler for each rtsp camera.
- It will then proccess all the streams in parallel in diferent threads. After we have got a new clip (We have reached a given number of frames) we would put that clip into a queue with a bunch of information like : 
    - clip_id;
    - camera_id;
    - frames;
    This object will be called ClipContainer or something;
- Then in another place we would access that queue and get the clips from there. Then we'll pass that clip to the FrameSampler. This component will -> get some N frames from the clip to be used down stream in the pipeline. The output from the framesampler will be : A SampledFrames object which will have the same metadata as the Clipcontainer and the sampled frames (Use the uniform sampling method for the clips). Maybe we could also just save the clip in disk in this phase if its the case (If the user is using a local disk storage for instance).

- The outputed SampledFrames will be then passed to the ObjectDetector component which will :
    - For each frame, detect all objects in it.
    - Then do tracking for each object and generate a trackedObjects or something Object;

- For each trackedobject we will:

    - generate an embedding for its tracked history using clip (useful to do behavior recognition + pure text retrieval);
    - generate an embedding just for the highest confidence for that object in that clip;
    - save the object information togather with its clip, camera and track id info into the milvus db. Here I should also include the timestamp information or clip path to disk. It will depend on the storage type the user chooses.