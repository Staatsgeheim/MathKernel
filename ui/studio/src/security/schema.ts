import { z } from 'zod';
// Disable runtime validator code generation in the application AND import worker.
z.config({ jitless: true });
export { z };
