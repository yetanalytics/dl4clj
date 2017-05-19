(ns datavec.api.split
  (:refer-clojure :exclude [reset!])
  (:import [org.datavec.api.split FileSplit BaseInputSplit CollectionInputSplit
            InputStreamInputSplit ListStringSplit NumberedFileInputSplit
            StringSplit TransformSplit TransformSplit$URITransform]
           [org.datavec.api.io.filters BalancedPathFilter RandomPathFilter]
           [org.datavec.api.io.labels PathLabelGenerator ParentPathLabelGenerator
            PatternPathLabelGenerator])
  (:require [clojure.java.io :as io]
            [dl4clj.nn.conf.utils :refer [contains-many?]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; input split multimethod
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn split-type
  "dispatch fn for input-split"
  [opts]
  (first (keys opts)))

(defmulti input-split
  "Multimethod that builds an input split based on the supplied type and opts"
  split-type)

(defmethod input-split :file-split [opts]
  (let [config (:file-split opts)
        {root :root-dir
         rng :rng-seed
         fmt :allow-format
         recursive? :recursive?} config]
    (cond (contains-many? config :root-dir :allow-format :rng-seed)
          (FileSplit. (io/as-file root) fmt rng)
          (contains-many? config :root-dir :allow-format :recursive?)
          (FileSplit. (io/as-file root) fmt recursive?)
          (contains-many? config :root-dir :allow-format)
          (FileSplit. (io/as-file root) fmt)
          (contains-many? config :root-dir :rng-seed)
          (FileSplit. (io/as-file root) rng)
          :else
          (FileSplit. (io/as-file root)))))

(defmethod input-split :collection-input-split [opts]
  (let [config (:collection-input-split opts)
        coll-of-uris (:collection config)]
    (CollectionInputSplit. coll-of-uris)))

(defmethod input-split :input-stream-input-split [opts]
  (let [config (:input-stream-input-split opts)
        {in :in-stream
         path :file-path} config]
    (if (contains? config :file-path)
      (InputStreamInputSplit. in path)
      (InputStreamInputSplit. in))))

(defmethod input-split :list-string-split [opts]
  (let [config (:list-string-split opts)
        d (:data config)]
    (ListStringSplit. d)))

(defmethod input-split :numbered-file-input-split [opts]
  (let [config (:numbered-file-input-split opts)
        {base-str :base-string
         min-idx :inclusive-min-idx
         max-idx :inclusive-max-idx} config]
    (NumberedFileInputSplit. base-str min-idx max-idx)))

(defmethod input-split :string-split [opts]
  (let [config (:string-split opts)
        d (:data config)]
    (StringSplit. d)))

(defmethod input-split :transform-split [opts]
  (let [config (:transform-split opts)
        {src-split :source-split
         t :transform
         u :uri} config]
    (TransformSplit. src-split (.apply t u))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; label generators
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-parent-path-label-generator
  "Returns as label the base name of the parent file of the path (the directory).

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/io/labels/ParentPathLabelGenerator.html"
  []
  (ParentPathLabelGenerator.))

(defn new-pattern-path-label-generator
  "Returns a label derived from the base name of the path.
  Splits the base name of the path with the given regex pattern,
  and returns the patternPosition'th element of the array.

  :pattern (str), a string regex
  :pattern-position (int), where to expect the patern within the file-path

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/io/labels/PatternPathLabelGenerator.html"
  [& {:keys [pattern pattern-position]
      :as opts}]
  (assert (contains? opts :pattern) "you must supply a regex patern to use this label generator")
  (if (contains? opts :pattern-position)
    (PatternPathLabelGenerator. pattern pattern-position)
    (PatternPathLabelGenerator. pattern)))

(defn get-label-for-path
  "used to infer the label of a file directly from the path of a file

  :label-generator (label-generator), call either new-parent-path-label-generator or
   new-pattern-path-label-generator
  :path (string or uri), the file path

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/io/labels/PathLabelGenerator.html"
  [& {:keys [label-generator path]}]
  (.getLabelForPath label-generator path))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; path-filter
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-balanced-path-filter
  "Randomizes the order of paths in an array and removes paths randomly to
  have the same number of paths for each label. Further interlaces the paths
  on output based on their labels, to obtain easily optimal batches for training.

  :rng (java.util.Random), a randomly generated number
  :extensions (array of strings), files to keep
  :label-generator (label-generator), call either new-parent-path-label-generator
   or new-pattern-path-label-generator
  :max-paths (int), max number of paths to return (0 = unlimited)
  :max-labels (int), max number of labels to return (0 = unlimited)
  :min-paths-per-label (int), min number of paths per labels to return
  :max-paths-per-label (int), max number of paths per labels to return
   - (0 = unlimited)
  :labels (collection of strings), file-paths you want to keep
   - empty collection = keep all paths

   If min-paths-per-label > 0, it might return an unbalanced set if the value is
  larger than the number of examples available for the label with the minimum amount.

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/io/filters/BalancedPathFilter.html"
  [& {:keys [rng extensions label-generator max-paths max-labels
             min-paths-per-label max-paths-per-label labels]
      :or {extensions nil
           max-paths 0
           max-labels 0
           min-paths-per-label 0
           max-paths-per-label 0}}]
  (BalancedPathFilter. rng extensions label-generator max-paths max-labels
                       min-paths-per-label max-paths-per-label labels))

(defn new-random-path-filter
  "Randomizes the order of paths in an array.

  :rng (java.util.Random), a randomly generated number
  :extensions (collection of strings), files to keep
  :max-paths (int), max number of paths to return
    - 0 = unlimited

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/io/filters/RandomPathFilter.html"
  [& {:keys [rng extensions max-paths]
      :or {max-paths 0}}]
  (RandomPathFilter. rng extensions max-paths))

(defn filter-paths
  ":path-filter (map), config opts for either balanced-path-filter or random-path-filter
  :paths (array of URIs), uri paths to be filtered"
  [& {:keys [path-filter paths]}]
  (.filter path-filter paths))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; input split user facing fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-filesplit
  "File input split. Splits up a root directory in to files.

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/split/FileSplit.html"
  [& {:keys [root-dir rng-seed allow-format recursive?]
      :as opts}]
  (input-split {:file-split opts}))

(defn new-input-stream-input-split
  "Input stream input split.
  The normal pattern is reading the whole input stream and turning that in to a record.
  This is meant for streaming raw data rather than normal mini batch pre processing.

  :file-path can be a file, a string, or a uri
  :in-stream, is the input stream

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/split/InputStreamInputSplit.html"
  [& {:keys [in-stream file-path]
      :as opts}]
  (input-split {:input-stream-input-split opts}))

(defn new-list-string-split
  "An input split that already has delimited data of some kind.

  :data should be a list of lists of strings

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/split/ListStringSplit.html"
  [& {:keys [data]
      :as opts}]
  (input-split {:list-string-split opts}))

(defn new-numbered-file-input-split
  "InputSplit for sequences of numbered files.

  ex. Suppose files are sequenced according to my_file_100.txt, my_file_101.txt,
      ..., my_file_200.txt  then use new-numbered-file-input-split as such:
      (new-numbered-file-input-split
       {:base-string my_file_%d.txt :inclusive-min-idx 100 :inclusive-max-idx 200})

  :base-string (str) regex for file names
  :inclusive-min-idx (int) starting index of files
  :inclusive-max-idx (int) end index of ffiles

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/split/NumberedFileInputSplit.html"
  [& {:keys [base-string inclusive-min-idx inclusive-max-idx]
      :as opts}]
  (input-split {:numbered-file-input-split opts}))

(defn new-string-split
  "String split used for single line inputs

  :data (str), a string to be used as data

   see: https://deeplearning4j.org/datavecdoc/org/datavec/api/split/StringSplit.html"
  [& {:keys [data]
      :as opts}]
  (input-split {:string-split opts}))

(defn new-transform-split
  "input-split implementation that maps the URIs of a given BaseInputSplit to new URIs.
  Useful when features and labels are in different files sharing a common naming scheme,
  and the name of the output file can be determined given the name of the input file.
   -Apply a given transformation to the raw URI objects

  :source-split (map), some form of input-split, ie. new-string-split, new-file-split ... (I believe)
   - java docs say base-input-split but its constructor is funky
  :transform (not sure), transform to apply to the uri, not sure of its type/form, will need testing
  :uri (java.net.URI), the uri to perform the transform on

  see: https://deeplearning4j.org/datavecdoc/org/datavec/api/split/TransformSplit.html"
  [& {:keys [source-split transform]
      :as opts}]
  (input-split {:transform-split opts}))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; input split api fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-root-dir
  [split]
  (.getRootDir split))

(defn get-is
  "return the input stream"
  [input-stream-input-split]
  (.getIs input-stream-input-split))

(defn set-is!
  "sets the input stream for an input-stream-input-split

  :is is a java.io.InputStream"
  [& {:keys [input-stream-input-split is]}]
  (doto input-stream-input-split (.setIs is)))

(defn initialize
  [split]
  (.initialize split))

(defn length
  "Length of the split"
  [split]
  (.length split))

(defn locations
  "Locations of the splits"
  [split]
  (.locations split))

(defn locations-iterator
  "returns the URI of the iterator assocaited with this filesplit"
  [split]
  (.locationsIterator split))

(defn locations-path-iterator
  "returns the file path of an iterator associated with this filesplit"
  [split]
  (.locationsPathIterator split))

(defn read-fields!
  "Deserialize the fields of this object from in."
  [& {:keys [split in]}]
  (doto split (.readFields in)))

(defn reset!
  "Reset the InputSplit without reinitializing it from scratch."
  [split]
  (doto split (.reset)))

(defn write!
  "Serialize the fields of this object to out."
  [& {:keys [split out-put-path]}]
  (doto split (.write out-put-path)))

(defn sample
  "Samples the locations based on the path-filter and splits
  the result into an array of input-pplit objects,
  with sizes proportional to the weights.

  args are:
  :path-filter (path-filter), call either new-balanced-path-filter or new-random-path-filter
   -to modify the locations (file paths) in some way (null == as is)
  :weights (double array) to split the locations into multiple InputSplit"
  [& {:keys [split path-filter weights-array]}]
  (.sample split path-filter weights-array))

(defn to-double!
  "convert writable to double"
  [writeable]
  (.toDouble writeable))

(defn to-float!
  "convert writeable to float"
  [writeable]
  (.toFloat writeable))

(defn to-int!
  "convert writeable to int"
  [writeable]
  (.toInt writeable))

(defn to-long!
  "convert writeable to long"
  [writeable]
  (.toLong writeable))
