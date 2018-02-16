// Common utilities for js scripts

const fs = require('fs-extra')
const yaml = require('js-yaml')
const path = require('path')
const zlib = require('zlib')

/**
 * Write array to gzipped numpy txt format
 */
const npSaveTxt = (array, filePath) => {
  let output = fs.createWriteStream(`${filePath}.gz`)
  let compress = zlib.createGzip()
  compress.pipe(output)

  for (let row of array) {
    let line
    if (Array.isArray(row)) {
      // This is a matrix
      line = row.map(it => Number.parseFloat(it).toExponential()).join(' ')
    } else {
      // This is a vector
      line = Number.parseFloat(row).toExponential()
    }
    compress.write(`${line}\n`)
  }
  compress.end()
}

/**
 * Return a list of paths to model directories
 */
const getModelDirs = (rootDir) => {
  let modelsDir = path.join(rootDir, 'model-forecasts/component-models')
  return fs.readdirSync(modelsDir)
    .map(it => path.join(modelsDir, it))
    .filter(it => fs.statSync(it).isDirectory())
    .filter(it => fs.existsSync(path.join(it, 'metadata.txt')))
}

/**
 * Read metadata files for given model
 */
const getModelMetadata = modelDir => {
  return readYamlFile(path.join(modelDir, 'metadata.txt'))
}

/**
 * Return model id from modelDir
 */
const getModelId = modelMeta => {
  return `${modelMeta.team_name}-${modelMeta.model_abbr}`
}

/**
 * Return timing information about the csv
 */
const getCsvTime = csvFile => {
  let baseName = path.basename(csvFile)
  let [epiweek, year, ] = baseName.split('-')
  return {
    epiweek: parseInt(epiweek.slice(2)) + '',
    year: year
  }
}

/**
 * Return path to all csvs in a model directory
 */
const getModelCsvs = modelDir => {
  return fs.readdirSync(modelDir)
    .filter(item => item.endsWith('csv'))
    .map(fileName => path.join(modelDir, fileName))
}

/**
 * Return unique items from array
 */
const unique = arr => {
  let hasNaN = false

  let uniqueItems = arr.reduce(function (acc, it) {
    if (Object.is(NaN, it)) {
      hasNaN = true
    } else if (acc.indexOf(it) === -1) {
      acc.push(it)
    }
    return acc
  }, [])

  return hasNaN ? [...uniqueItems, NaN] : uniqueItems
}

const readYamlFile = fileName => {
  return yaml.safeLoad(fs.readFileSync(fileName, 'utf8'))
}

module.exports.unique = unique
module.exports.readYamlFile = readYamlFile
module.exports.getModelDirs = getModelDirs
module.exports.getModelMetadata = getModelMetadata
module.exports.getModelId = getModelId
module.exports.getCsvTime = getCsvTime
module.exports.getModelCsvs = getModelCsvs
module.exports.npSaveTxt = npSaveTxt
