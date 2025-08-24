/**
 * File Upload Component with Validation and Progress
 */

import React, { useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Alert, AlertDescription } from '../ui/alert';
import { Progress } from '../ui/progress';
import { Badge } from '../ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '../ui/dialog';

import { cn } from '../../lib/utils';
import {
  CloudUpload,
  CheckCircle,
  AlertCircle,
  AlertTriangle,
  FileText,
  BarChart3,
  Gauge,
} from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { useUploadData, useValidateData } from '../../hooks/useAPI';

const FileUpload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [validationResult, setValidationResult] = useState<any>(null);
  const [showUploadDialog, setShowUploadDialog] = useState(false);

  const uploadMutation = useUploadData();
  const validateMutation = useValidateData();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setValidationResult(null);
      
      // Auto-validate the file
      validateMutation.mutate(file, {
        onSuccess: (result) => {
          setValidationResult(result);
        },
        onError: (error) => {
          console.error('Validation failed:', error);
        },
      });
    }
  }, [validateMutation]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'text/csv': ['.csv'],
    },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024, // 50MB
  });

  const handleUpload = () => {
    if (!selectedFile) return;
    
    uploadMutation.mutate(selectedFile, {
      onSuccess: (result) => {
        setShowUploadDialog(false);
        setSelectedFile(null);
        setValidationResult(null);
      },
      onError: (error) => {
        console.error('Upload failed:', error);
      },
    });
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };


  
  const getQualityColor = (score: number) => {
    if (score >= 90) return 'success';
    if (score >= 70) return 'warning';
    return 'error';
  };

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Upload Flight Data</CardTitle>
          <p className="text-sm text-muted-foreground">
            Upload Excel (.xlsx) or CSV files containing flight schedule data
          </p>
        </CardHeader>
        <CardContent>
          {/* Drop Zone */}
          <div
            {...getRootProps()}
            className={cn(
              "border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-300",
              isDragActive 
                ? "border-primary bg-primary/5" 
                : "border-muted-foreground/25 hover:border-primary hover:bg-muted/50"
            )}
          >
            <input {...getInputProps()} />
            <CloudUpload className="h-12 w-12 text-primary mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">
              {isDragActive ? 'Drop file here' : 'Drag & drop file here'}
            </h3>
            <p className="text-sm text-muted-foreground mb-2">
              or click to select file
            </p>
            <p className="text-xs text-muted-foreground">
              Supported formats: .xlsx, .xls, .csv (max 50MB)
            </p>
          </div>

          {/* File Info */}
          {selectedFile && (
            <div className="mt-6 space-y-4">
              <Alert>
                <AlertDescription>
                  <div className="flex items-center gap-2">
                    <FileText className="h-4 w-4" />
                    <span>
                      <strong>{selectedFile.name}</strong> ({formatFileSize(selectedFile.size)})
                    </span>
                  </div>
                </AlertDescription>
              </Alert>

              {/* Validation Progress */}
              {validateMutation.isPending && (
                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">
                    Validating file...
                  </p>
                  <Progress value={undefined} className="w-full" />
                </div>
              )}

              {/* Validation Results */}
              {validationResult && (
                <Card className="border">
                  <CardContent className="p-4">
                    <div className="flex items-center gap-2 mb-4">
                      <Badge 
                        variant="outline" 
                        className={cn(
                          validationResult.validation.valid 
                            ? 'border-green-500 text-green-700' 
                            : 'border-red-500 text-red-700'
                        )}
                      >
                        {validationResult.validation.valid ? (
                          <CheckCircle className="mr-1 h-3 w-3" />
                        ) : (
                          <AlertCircle className="mr-1 h-3 w-3" />
                        )}
                        {validationResult.validation.valid ? 'Valid' : 'Invalid'}
                      </Badge>
                      <Badge variant="outline">
                        {validationResult.validation.file_info?.rows || 0} rows
                      </Badge>
                      <Badge variant="outline">
                        {validationResult.validation.file_info.columns?.length || 0} columns
                      </Badge>
                    </div>

                    {/* Errors */}
                    {validationResult.validation.errors && validationResult.validation.errors.length > 0 && (
                      <Alert className="border-destructive mb-4">
                        <AlertDescription>
                          <p className="font-semibold mb-2">
                            Validation Errors:
                          </p>
                          <div className="space-y-2">
                            {(validationResult.validation.errors || []).map((error: string, index: number) => (
                              <div key={index} className="flex items-start gap-2">
                                <AlertCircle className="h-4 w-4 text-destructive mt-0.5 flex-shrink-0" />
                                <span className="text-sm">{error}</span>
                              </div>
                            ))}
                          </div>
                        </AlertDescription>
                      </Alert>
                    )}

                    {/* Warnings */}
                    {validationResult.validation.warnings && validationResult.validation.warnings.length > 0 && (
                      <Alert className="border-yellow-500 mb-4">
                        <AlertDescription>
                          <p className="font-semibold mb-2">
                            Warnings:
                          </p>
                          <div className="space-y-2">
                            {(validationResult.validation.warnings || []).map((warning: string, index: number) => (
                              <div key={index} className="flex items-start gap-2">
                                <AlertTriangle className="h-4 w-4 text-yellow-600 mt-0.5 flex-shrink-0" />
                                <span className="text-sm">{warning}</span>
                              </div>
                            ))}
                          </div>
                        </AlertDescription>
                      </Alert>
                    )}

                    {/* Recommendations */}
                    {validationResult.recommendations.length > 0 && (
                      <div>
                        <p className="font-semibold mb-2">
                          Recommendations:
                        </p>
                        <div className="space-y-2">
                          {validationResult.recommendations.map((rec: string, index: number) => (
                            <div key={index} className="flex items-start gap-2">
                              <BarChart3 className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                              <span className="text-sm">{rec}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}

              {/* Upload Button */}
              {validationResult?.validation.valid && (
                <div className="flex justify-center">
                  <Button
                    onClick={() => setShowUploadDialog(true)}
                    disabled={uploadMutation.isPending}
                    size="lg"
                    className="gap-2"
                  >
                    <CloudUpload className="h-4 w-4" />
                    Process & Upload Data
                  </Button>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Upload Confirmation Dialog */}
      <Dialog open={showUploadDialog} onOpenChange={setShowUploadDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Confirm Data Upload</DialogTitle>
          </DialogHeader>
          {selectedFile && validationResult && (
            <div className="space-y-4">
              <p className="text-sm">
                Ready to process and upload: <strong>{selectedFile.name}</strong>
              </p>
              
              <div>
                <p className="text-sm font-semibold mb-2">
                  File Details:
                </p>
                <div className="flex gap-2 flex-wrap">
                  <Badge variant="outline">{validationResult.validation.file_info?.rows || 0} flights</Badge>
                  <Badge variant="outline">{validationResult.validation.file_info.columns?.length || 0} columns</Badge>
                  <Badge variant="outline">{formatFileSize(selectedFile.size)}</Badge>
                </div>
              </div>

              <Alert>
                <AlertDescription>
                  <p className="font-semibold mb-2">The upload will:</p>
                  <ul className="space-y-1 text-sm">
                    <li>• Process and validate all flight data</li>
                    <li>• Store data in PostgreSQL database</li>
                    <li>• Retrain ML models with new data</li>
                    <li>• Update analytics and predictions</li>
                  </ul>
                </AlertDescription>
              </Alert>

              {uploadMutation.isPending && (
                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">
                    Processing upload...
                  </p>
                  <Progress value={undefined} className="w-full" />
                </div>
              )}

              {uploadMutation.error && (
                <Alert className="border-destructive">
                  <AlertDescription>
                    Upload failed: {uploadMutation.error.message}
                  </AlertDescription>
                </Alert>
              )}

              {uploadMutation.data && (
                <Alert className="border-green-500">
                  <AlertDescription>
                    <p className="font-semibold mb-2">
                      Upload Successful!
                    </p>
                    <div className="flex gap-2 flex-wrap">
                      <Badge variant="outline" className="border-green-500 text-green-700">
                        <CheckCircle className="mr-1 h-3 w-3" />
                        {uploadMutation.data.processing_result.records_processed} records processed
                      </Badge>
                      <Badge 
                        variant="outline" 
                        className={cn(
                          getQualityColor(uploadMutation.data.processing_result.data_quality_score || 0) === 'success' ? 'border-green-500 text-green-700' :
                          getQualityColor(uploadMutation.data.processing_result.data_quality_score || 0) === 'warning' ? 'border-yellow-500 text-yellow-700' :
                          'border-red-500 text-red-700'
                        )}
                      >
                        <Gauge className="mr-1 h-3 w-3" />
                        Quality: {uploadMutation.data.processing_result.data_quality_score?.toFixed(1)}%
                      </Badge>
                    </div>
                  </AlertDescription>
                </Alert>
              )}
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowUploadDialog(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleUpload}
              disabled={uploadMutation.isPending || uploadMutation.isSuccess}
            >
              {uploadMutation.isPending ? 'Processing...' : 'Confirm Upload'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default FileUpload;
