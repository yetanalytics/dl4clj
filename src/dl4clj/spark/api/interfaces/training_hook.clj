(ns ^{:doc "A hook for the workers when training. A pre update and post update method are specified for when certain information needs to be collected or there needs to be specific parameters or models sent to remote locations for visualization or other things.
see: https://deeplearning4j.org/doc/org/deeplearning4j/spark/api/TrainingHook.html"}
    dl4clj.spark.api.interfaces.training-hook
  (:import [org.deeplearning4j.spark.api TrainingHook]))

;; no implemented training hook classes If i remember correctly
;; this should be worth keeping around



;; come back to this now that 0.9.0 is a thing
(defn post-update!
  "A hook method for post update

  :mini-batch (dataset), the mini-batch of a larger dataset, can be a dataset or a multi-dataset
   - see: nd4clj.linalg.dataset.api.data-set and nd4clj.linalg.dataset.data-set

  :model (model), a nn or layer which implements the model interface.

  :training-hook (hook), the training hook used for the update
   - see: ... (none are in 0.8.0), will have to wait for next release or implement myself via gen-class

  returns a map of the supplied args"
  [& {:keys [training-hook mini-batch model]}]
  (.postUpdate training-hook mini-batch model)
  {:this training-hook
   :ds mini-batch
   :model model})

(defn pre-update!
  "A hook method for pre update

  :mini-batch (dataset), the mini-batch of a larger dataset, can be a dataset or a multi-dataset
   - see: nd4clj.linalg.dataset.api.data-set and nd4clj.linalg.dataset.data-set

  :model (model), a nn or layer which implements the model interface.

  :training-hook (hook), the training hook used for the update
   - see: ... (none are in 0.8.0), will have to wait for next release or implement myself via gen-class

  returns a map of the supplied args"
  [& {:keys [training-hook mini-batch model]}]
  (.preUpdate training-hook mini-batch model)
  {:this training-hook
   :ds mini-batch
   :model model})
