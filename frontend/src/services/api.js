import axios from 'axios';

const API_BASE = '/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const inspectionService = {
  // Upload video or image
  uploadFile: async (file, onProgress) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/inspection/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(percent);
        }
      },
    });
  },

  // Start processing
  startProcessing: (inspectionId, config) => {
    return api.post(`/inspection/${inspectionId}/process`, config);
  },

  // Poll status
  getStatus: (inspectionId) => {
    return api.get(`/inspection/${inspectionId}/status`);
  },

  // Get full inspection results
  getResults: (inspectionId) => {
    return api.get(`/inspection/${inspectionId}/results`);
  },

  // Get frame image links
  getFrames: (inspectionId) => {
    return api.get(`/inspection/${inspectionId}/frames`);
  },

  // Get history
  getHistory: (limit = 20) => {
    return api.get('/inspection/history', { params: { limit } });
  },

  // Download reports
  getReportDownloadUrl: (inspectionId, format = 'pdf') => {
    return `/api/reports/${inspectionId}/download?format=${format}`;
  },

  // Video URL
  getVideoUrl: (inspectionId) => {
    return `/api/inspection/${inspectionId}/video`;
  },
};

export const modelService = {
  getStatus: () => api.get('/models/status'),
  getDatasetInfo: () => api.get('/models/dataset'),
  trainModel: (config) => api.post('/models/train', config),
  getTrainingStatus: () => api.get('/models/train/status'),
  validateModel: (params) => api.post('/models/validate', null, { params }),
};

export default api;
