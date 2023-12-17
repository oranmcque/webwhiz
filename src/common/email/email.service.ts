import { Injectable, Logger } from '@nestjs/common';
import * as sgMail from '@sendgrid/mail';
import { AppConfigService } from '../config/appConfig.service';

@Injectable()
export class EmailService {
  private readonly logger: Logger;
  private isSgInitialized: boolean;

  constructor(private appConfig: AppConfigService) {
    this.logger = new Logger(EmailService.name);

    const sendGridApiKey = this.appConfig.get('sendGridApiKey');
    if (!sendGridApiKey) {
      this.isSgInitialized = false;
    } else {
      sgMail.setApiKey(sendGridApiKey);
      this.isSgInitialized = true;
    }
  }

  async sendWelcomeEmail(email: string, username?: string) {
    if (!this.isSgInitialized) return;

    const msg = {
      to: email,
      from: { email: 'hi@webwhiz.ai', name: 'WebWhiz.ai' },
      templateId: 'd-4ba3b2bf1f9e4fcea8c5e881c934a2c6',
      dynamicTemplateData: {
        username,
      },
    };

    const res = await sgMail.send(msg);
    return res;
  }

  async sendEmailWithAttachment(
    email: string,
    attachmentName: string,
    attachmentType: string,
    attachment: Buffer,
  ) {
    const msg = {
      to: email,
      from: { email: 'hi@webwhiz.ai', name: 'WebWhiz.ai' },
      subject: 'Sending with SendGrid is Fun',
      text: 'Hi,\n\nPFA the website crawl data',
      attachments: [
        {
          content: attachment.toString('base64'),
          filename: attachmentName,
          type: attachmentType,
          disposition: 'attachment',
        },
      ],
    };

    const res = await sgMail.send(msg);
    return res;
  }

  async sendOfflineMsgEmail(
    email: string,
    websiteUrl: string,
    queryEmail: string,
    queryText: string,
    queryName?: string,
  ) {
    if (!this.isSgInitialized) return;

    const msg = {
      to: email,
      from: { email: 'hi@webwhiz.ai', name: 'WebWhiz.ai' },
      replyTo: queryEmail,
      templateId: 'd-a4f34375b9504c5d94f6d9e3eafe0214',
      dynamicTemplateData: {
        website_url: websiteUrl,
        msg_email: queryEmail,
        msg_name: queryName || 'NA',
        msg_msg: queryText,
      },
    };

    const res = await sgMail.send(msg);
    return res;
  }

  async sendToken80ExhaustedEmail(email: string) {
    if (!this.isSgInitialized) return;

    const msg = {
      to: email,
      from: { email: 'hi@webwhiz.ai', name: 'WebWhiz.ai' },
      templateId: 'd-8bcb4db974004aa080a4c06e011ddc8a',
    };

    const res = await sgMail.send(msg);
    return res;
  }

  async sendToken100ExhaustedEmail(email: string) {
    if (!this.isSgInitialized) return;

    const msg = {
      to: email,
      from: { email: 'hi@webwhiz.ai', name: 'WebWhiz.ai' },
      templateId: 'd-7d140563825745a2b4c8e02afab18aea',
    };

    const res = await sgMail.send(msg);
    return res;
  }
}
