(ns ^{:doc "ns for saving models in an early training context"}
    dl4clj.earlystopping.model-saver
  (:import [org.deeplearning4j.earlystopping.saver
            InMemoryModelSaver
            LocalFileModelSaver]
           [java.nio.charset Charset])
  (:require [dl4clj.utils :refer [generic-dispatching-fn obj-or-code?]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multimethods for calling the dl4j constructors
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti model-saver-type generic-dispatching-fn)

(defmethod model-saver-type :in-memory [opts]
  `(InMemoryModelSaver.))

(defmethod model-saver-type :local-file [opts]
  (let [config (:local-file opts)
        {dir :directory
         char-set :charset} config]
    (if char-set
      `(LocalFileModelSaver. ~dir ~char-set)
      `(LocalFileModelSaver. ~dir))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns for creating model savers
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-in-memory-saver
  "Save the best (and latest) models for early stopping training to memory for later retrieval

  :as-code? (boolean), return java object or code for creating it

  see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/saver/InMemoryModelSaver.html"
  [& {:keys [as-code?]
      :or {as-code? true}}]
  (let [code (model-saver-type {:in-memory {}})]
    (obj-or-code? as-code? code)))

(defn new-local-file-model-saver
  "Save the best (and latest/most recent) models learned during
  early stopping training to the local file system.

  Instances of this class will save 3 files for best (and optionally, latest) models:
   (a) The network configuration: bestModelConf.json
   (b) The network parameters: bestModelParams.bin
   (c) The network updater: bestModelUpdater.bin
    - not needed during testing but is kept around so future training is possible.
      - holds the interal state of the model

  :directory (str), the directory to save in

  :charset (charset), a java character set.
   - for avialable char sets, eval (Charset/availableCharsets)

  :as-code? (boolean), return java object or code for creating it

  see: https://deeplearning4j.org/doc/org/deeplearning4j/earlystopping/saver/LocalFileModelSaver.html"
  [& {:keys [directory charset as-code?]
      :or {as-code? true}
      :as opts}]
  (let [code (model-saver-type {:local-file opts})]
    (obj-or-code? as-code? code)))
